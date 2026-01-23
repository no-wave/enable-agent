#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iris Classification RAG 챗봇 모듈

Enable Agent와 Context Builder를 통합하여 대화형 인터페이스를 제공한다.
"""

import json
from typing import Dict, Any, List, Optional

from openai import OpenAI

from .enable_agent import IrisClassificationAgent
from .context_builder import PredictionContextBuilder


class IrisRAGChatbot:
    """Iris 분류 RAG 챗봇"""
    
    def __init__(self, agent: IrisClassificationAgent, context_builder: PredictionContextBuilder):
        """
        챗봇을 초기화한다.
        
        Args:
            agent: IrisClassificationAgent 인스턴스
            context_builder: PredictionContextBuilder 인스턴스
        """
        self.agent = agent
        self.context_builder = context_builder
        self.client = OpenAI()
        self.conversation_history: List[Dict] = []
        
        print("✓ RAG 챗봇 초기화 완료")
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트를 생성한다."""
        summary = self.context_builder.get_summary()
        knowledge_base = self.context_builder.get_knowledge_base_content()
        
        # 지식 베이스가 너무 길면 요약만 사용
        if len(knowledge_base) > 3000:
            knowledge_base = knowledge_base[:3000] + "\n...(생략)..."
        
        feature_importance = self.agent.get_feature_importance()
        
        return f"""당신은 붓꽃(Iris) 품종 분류 전문 AI 어시스턴트다.
Random Forest 기반 분류 모델을 사용하여 붓꽃 품종을 예측하고 설명한다.

역할:
1. 입력된 꽃잎/꽃받침 특성으로 붓꽃 품종(Setosa, Versicolor, Virginica)을 예측한다
2. 예측 결과에 대한 설명을 제공한다
3. 붓꽃 품종에 대한 정보를 안내한다
4. 예측 통계와 분석 결과를 제공한다

현재 통계:
- 총 예측 횟수: {summary.get('total_predictions', 0)}회
- 평균 신뢰도: {summary.get('average_confidence', 0):.1%}
- 가장 많이 예측된 품종: {summary.get('most_common_class', 'N/A')}

클래스별 분포: {json.dumps(summary.get('class_distribution', {}), ensure_ascii=False)}

특성 중요도: {json.dumps(feature_importance, ensure_ascii=False)}

지식 베이스:
{knowledge_base}

응답 지침:
- 문장은 ~다로 끝낸다
- 붓꽃 관련 질문에는 정확한 정보를 제공한다
- 예측 요청 시 predict 도구를 사용한다
- 특성 값이 주어지면 품종을 예측한다
"""
    
    def chat(self, user_message: str) -> str:
        """
        사용자 메시지에 응답한다.
        
        Args:
            user_message: 사용자 메시지
        
        Returns:
            str: 챗봇 응답
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # 시스템 프롬프트 + 대화 히스토리
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        messages.extend(self.conversation_history)
        
        # 도구 정의
        tools = [self.agent.generate_tool_definition()]
        
        # OpenAI 호출
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        # Tool Call 처리
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                if function_name == 'predict':
                    # 예측 실행
                    result = self.agent.predict(arguments)
                    
                    # 결과를 Context에 저장
                    self.context_builder.add_prediction(result)
                    
                    # 대화 히스토리에 추가
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result, ensure_ascii=False, default=str)
                    })
            
            # 최종 응답 생성
            final_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": self._get_system_prompt()}] + self.conversation_history
            )
            
            final_message = final_response.choices[0].message.content
        else:
            final_message = assistant_message.content
        
        self.conversation_history.append({
            "role": "assistant",
            "content": final_message
        })
        
        return final_message
    
    def predict_and_explain(self, features: Dict) -> str:
        """
        예측을 수행하고 결과를 설명한다.
        
        Args:
            features: 입력 특성
        
        Returns:
            str: 예측 결과 및 설명
        """
        # 예측 수행
        result = self.agent.predict(features)
        
        # 결과 저장
        self.context_builder.add_prediction(result)
        
        # 설명 생성
        explanation = self.agent.generate_explanation(result)
        
        response = f"""=== 예측 결과 ===
품종: {result['predicted_class']} ({result['korean_name']})
신뢰도: {result['confidence']:.1%}

=== 클래스별 확률 ===
- Setosa: {result['probabilities']['setosa']:.1%}
- Versicolor: {result['probabilities']['versicolor']:.1%}
- Virginica: {result['probabilities']['virginica']:.1%}

=== 설명 ===
{explanation}
"""
        return response
    
    def generate_report(self) -> str:
        """분석 보고서를 생성한다."""
        summary = self.context_builder.get_summary()
        class_stats = self.context_builder.get_class_statistics()
        
        prompt = f"""붓꽃 분류 분석 보고서를 작성해달라.

현재까지의 분석 결과:
- 총 예측 횟수: {summary.get('total_predictions', 0)}회
- 평균 신뢰도: {summary.get('average_confidence', 0):.1%}
- 가장 많이 예측된 품종: {summary.get('most_common_class', 'N/A')} ({summary.get('most_common_count', 0)}회)

클래스별 분포: {summary.get('class_distribution', {})}

클래스별 통계: {json.dumps(class_stats, ensure_ascii=False, default=str)}

특성 중요도: {self.agent.get_feature_importance()}

다음 내용을 포함하여 보고서를 작성하라:
1. 분석 개요
2. 클래스별 예측 현황
3. 모델 성능 인사이트
4. 특성 분석

문장은 ~다로 끝낸다."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    def reset_conversation(self):
        """대화 히스토리를 초기화한다."""
        self.conversation_history = []
        print("✓ 대화 히스토리 초기화 완료")


def interactive_chat(agent: IrisClassificationAgent, context_builder: PredictionContextBuilder):
    """대화형 챗봇을 실행한다."""
    chatbot = IrisRAGChatbot(agent, context_builder)
    
    print("\n" + "=" * 60)
    print("Iris 분류 챗봇")
    print("=" * 60)
    print("명령어:")
    print("  - 종료: 'quit' 또는 'exit'")
    print("  - 보고서: 'report'")
    print("  - 대화 초기화: 'reset'")
    print("예시 질문:")
    print("  - '꽃잎 길이 4.5cm, 꽃잎 너비 1.5cm, 꽃받침 길이 6cm, 너비 3cm인 붓꽃은?'")
    print("  - 'setosa 품종의 특징은?'")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("사용자: ").strip()
        except (UnicodeDecodeError, UnicodeError):
            print("[WARNING] 입력 인코딩 오류, 다시 입력해달라.")
            continue
        except (EOFError, KeyboardInterrupt):
            print("\n챗봇을 종료한다.")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("챗봇을 종료한다.")
            break
        
        if user_input.lower() == 'report':
            print("\n보고서 생성 중...")
            report = chatbot.generate_report()
            print("\n" + "=" * 60)
            print("분석 보고서")
            print("=" * 60)
            print(report)
            print("=" * 60 + "\n")
            continue
        
        if user_input.lower() == 'reset':
            chatbot.reset_conversation()
            continue
        
        try:
            response = chatbot.chat(user_input)
            print(f"\n챗봇: {response}\n")
        except Exception as e:
            print(f"\n[ERROR] 응답 생성 실패: {e}\n")
    
    return chatbot


if __name__ == '__main__':
    from .enable_agent import IrisClassificationAgent, create_skill_file
    
    # Agent 설정
    skill_path = create_skill_file()
    agent = IrisClassificationAgent(skill_path)
    
    # Context Builder 설정
    context_builder = PredictionContextBuilder()
    
    # 대화형 챗봇 실행
    interactive_chat(agent, context_builder)
