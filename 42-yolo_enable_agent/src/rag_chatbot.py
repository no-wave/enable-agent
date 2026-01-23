#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Plastic Detection RAG 챗봇 모듈

Enable Agent와 Context Builder를 통합하여 대화형 인터페이스를 제공한다.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from openai import OpenAI

from .enable_agent import PlasticDetectionAgent
from .context_builder import DetectionContextBuilder


class PlasticRAGChatbot:
    """플라스틱 탐지 RAG 챗봇"""
    
    def __init__(self, agent: PlasticDetectionAgent, context_builder: DetectionContextBuilder):
        """
        챗봇을 초기화한다.
        
        Args:
            agent: PlasticDetectionAgent 인스턴스
            context_builder: DetectionContextBuilder 인스턴스
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
        
        return f"""당신은 플라스틱 재활용 분류 전문 AI 어시스턴트다.
YOLOv8 기반 객체 탐지 모델을 사용하여 플라스틱을 분류하고 재활용 가이드를 제공한다.

역할:
1. 이미지에서 플라스틱 객체(PET, PS, PP, PE)를 탐지한다
2. 탐지된 플라스틱의 재활용 가능 여부를 판단한다
3. 올바른 분리수거 방법을 안내한다
4. 탐지 통계와 분석 결과를 제공한다

현재 통계:
- 총 분석 이미지: {summary.get('total_images', 0)}개
- 총 탐지 객체: {summary.get('total_detections', 0)}개
- 재활용 가능: {summary.get('total_recyclable', 0)}개
- 재활용 불가: {summary.get('total_non_recyclable', 0)}개
- 평균 탐지 신뢰도: {summary.get('average_confidence', 0):.1%}

클래스별 분포: {json.dumps(summary.get('class_distribution', {}), ensure_ascii=False)}

지식 베이스:
{knowledge_base}

응답 지침:
- 문장은 ~다로 끝낸다
- 재활용 관련 질문에는 정확한 정보를 제공한다
- 탐지 결과 분석 요청 시 통계를 활용한다
- 이미지 분석 요청 시 detect 도구를 사용한다
"""
    
    def chat(self, user_message: str, image_path: Optional[str] = None) -> str:
        """
        사용자 메시지에 응답한다.
        
        Args:
            user_message: 사용자 메시지
            image_path: 분석할 이미지 경로 (선택)
        
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
                
                # 이미지 경로가 제공되면 사용
                if image_path and 'image_path' not in arguments:
                    arguments['image_path'] = image_path
                
                if function_name == 'detect':
                    # 탐지 실행
                    result = self.agent.detect(**arguments)
                    
                    # 결과를 Context에 저장
                    self.context_builder.add_detection(result)
                    
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
    
    def analyze_image(self, image_path: str) -> str:
        """
        이미지를 분석하고 결과를 설명한다.
        
        Args:
            image_path: 이미지 파일 경로
        
        Returns:
            str: 분석 결과 설명
        """
        # 탐지 실행
        result = self.agent.detect(image_path)
        
        # 결과 저장
        self.context_builder.add_detection(result)
        
        if not result.get('detections'):
            return "이미지에서 플라스틱 객체가 탐지되지 않았다."
        
        # 결과 설명 생성
        explanation = self.agent.generate_explanation(result)
        guide = self.agent.get_recycling_guide(result)
        
        return f"{explanation}\n\n{guide}"
    
    def generate_report(self) -> str:
        """분석 보고서를 생성한다."""
        summary = self.context_builder.get_summary()
        class_stats = self.context_builder.get_class_statistics()
        
        prompt = f"""플라스틱 탐지 분석 보고서를 작성해달라.

현재까지의 분석 결과:
- 총 분석 이미지: {summary.get('total_images', 0)}개
- 총 탐지 객체: {summary.get('total_detections', 0)}개
- 재활용 가능: {summary.get('total_recyclable', 0)}개 ({summary.get('recyclable_ratio', 0):.1%})
- 재활용 불가: {summary.get('total_non_recyclable', 0)}개

클래스별 통계: {json.dumps(class_stats, ensure_ascii=False, default=str)}

다음 내용을 포함하여 보고서를 작성하라:
1. 분석 개요
2. 클래스별 탐지 현황
3. 재활용 관련 인사이트
4. 개선 제안

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


def interactive_chat(agent: PlasticDetectionAgent, context_builder: DetectionContextBuilder):
    """대화형 챗봇을 실행한다."""
    chatbot = PlasticRAGChatbot(agent, context_builder)
    
    print("\n" + "=" * 60)
    print("플라스틱 재활용 분류 챗봇")
    print("=" * 60)
    print("명령어:")
    print("  - 종료: 'quit' 또는 'exit'")
    print("  - 보고서: 'report'")
    print("  - 이미지 분석: 'analyze <이미지경로>'")
    print("  - 대화 초기화: 'reset'")
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
        
        if user_input.lower().startswith('analyze '):
            image_path = user_input[8:].strip()
            if Path(image_path).exists():
                print(f"\n이미지 분석 중: {image_path}")
                result = chatbot.analyze_image(image_path)
                print(f"\n챗봇: {result}\n")
            else:
                print(f"\n이미지를 찾을 수 없다: {image_path}\n")
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
    from .enable_agent import PlasticDetectionAgent, create_skill_file
    
    # Agent 설정
    skill_path = create_skill_file()
    agent = PlasticDetectionAgent(skill_path)
    
    # Context Builder 설정
    context_builder = DetectionContextBuilder()
    
    # 대화형 챗봇 실행
    interactive_chat(agent, context_builder)
