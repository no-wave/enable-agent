"""
RAG 챗봇 모듈

지식 베이스를 활용하여 대화형 예측 인터페이스를 제공한다.
"""

import json
from typing import Dict, Any, List

from openai import OpenAI

from .enable_agent import SpaceshipEnableAgent
from .context_builder import PredictionContextBuilder


class SpaceshipRAGChatbot:
    """Spaceship Titanic 예측 지식을 활용하는 RAG 챗봇"""
    
    def __init__(self, agent: SpaceshipEnableAgent, context_builder: PredictionContextBuilder):
        """RAG 챗봇을 초기화한다."""
        self.agent = agent
        self.context_builder = context_builder
        self.client = OpenAI()
        self.conversation_history: List[Dict[str, Any]] = []
        
        print("RAG 챗봇 초기화 완료")
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트를 생성한다."""
        knowledge_base = self.context_builder.get_knowledge_base_content()
        agent_capability = self.agent.get_capability_description()
        
        return f"""
당신은 Spaceship Titanic 승객의 다른 차원 이동 여부를 예측하는 AI 어시스턴트다.
다음과 같은 역할을 수행한다:

1. 사용자의 승객 정보를 받아 이동 여부를 예측한다
2. 과거 예측 기록을 바탕으로 통계와 인사이트를 제공한다
3. Spaceship Titanic의 배경과 특성에 대해 설명한다
4. 예측 결과에 대한 상세한 분석과 해석을 제공한다

## 사용 가능한 도구

{agent_capability}

## 과거 예측 지식 베이스

{knowledge_base}

## 응답 가이드라인

- 친절하고 전문적인 톤을 유지한다
- 기술 용어를 사용할 때는 쉬운 설명을 덧붙인다
- 과거 예측 데이터를 활용하여 맥락있는 답변을 제공한다
- 예측 요청 시 predict_spaceship_transport 함수를 호출한다
- 문장의 끝은 ~다로 끝낸다
""".strip()
    
    def chat(self, user_message: str) -> str:
        """사용자 메시지에 응답한다."""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        messages = [{"role": "system", "content": self._get_system_prompt()}] + self.conversation_history
        tools = [self.agent.generate_tool_definition()]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7
        )
        
        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "predict_spaceship_transport":
                    prediction_result = self.agent.predict(function_args)
                    self.context_builder.add_prediction(prediction_result)
                    
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(prediction_result, ensure_ascii=False)
                    })
            
            messages = [{"role": "system", "content": self._get_system_prompt()}] + self.conversation_history
            
            final_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7
            )
            
            assistant_message = final_response.choices[0].message.content
        else:
            assistant_message = response_message.content
        
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def generate_report(self) -> str:
        """분석 보고서를 생성한다."""
        summary = self.context_builder.get_summary()
        knowledge_base = self.context_builder.get_knowledge_base_content()
        statistics = self.context_builder.get_statistics()
        
        messages = [
            {
                "role": "system",
                "content": "당신은 데이터 분석 보고서를 작성하는 전문가다. 주어진 예측 데이터를 분석하여 경영진이 이해하기 쉬운 보고서를 작성한다. 문장의 끝은 ~다로 끝낸다."
            },
            {
                "role": "user",
                "content": f"""
다음 Spaceship Titanic 예측 데이터를 바탕으로 상세한 분석 보고서를 작성해달라:

## 요약 통계
{json.dumps(summary, indent=2, ensure_ascii=False)}

## 상세 통계
{json.dumps(statistics, indent=2, ensure_ascii=False)}

## 예측 기록
{knowledge_base}

보고서에 다음 내용을 포함해달라:
1. 전체 예측 성능 요약
2. 이동/비이동 패턴 분석
3. 주요 영향 요인 분석 (냉동 수면, 지출 패턴 등)
4. 인사이트 및 권장사항
5. 향후 개선 방향
"""
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def reset_conversation(self):
        """대화 히스토리를 초기화한다."""
        self.conversation_history = []
        print("대화 히스토리 초기화 완료")
    
    def get_conversation_summary(self) -> str:
        """현재 대화 요약을 반환한다."""
        if not self.conversation_history:
            return "대화 기록이 없다."
        
        user_messages = [m['content'] for m in self.conversation_history if m['role'] == 'user']
        return f"총 {len(user_messages)}개의 사용자 메시지가 있다."


def interactive_chat(agent: SpaceshipEnableAgent, context_builder: PredictionContextBuilder):
    """인터랙티브 챗봇 세션을 시작한다."""
    chatbot = SpaceshipRAGChatbot(agent, context_builder)
    
    print("\n" + "=" * 60)
    print("Spaceship Titanic RAG 챗봇")
    print("=" * 60)
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("보고서 생성: 'report'")
    print("대화 초기화: 'reset'")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("사용자: ").strip()
        
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
        
        response = chatbot.chat(user_input)
        print(f"\n챗봇: {response}\n")
    
    return chatbot


if __name__ == '__main__':
    from .enable_agent import SpaceshipEnableAgent
    from .context_builder import PredictionContextBuilder
    
    agent = SpaceshipEnableAgent('skills/agent_skill.yaml')
    context_builder = PredictionContextBuilder()
    
    interactive_chat(agent, context_builder)
