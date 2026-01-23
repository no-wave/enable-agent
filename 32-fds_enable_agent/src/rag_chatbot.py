"""
FDS RAG 챗봇 모듈

지식 베이스를 활용하여 대화형 사기 탐지 인터페이스를 제공한다.
"""

import json
from typing import Dict, Any, List

from openai import OpenAI

from .enable_agent import FDSEnableAgent
from .context_builder import FDSContextBuilder


class FDSRAGChatbot:
    """FDS 지식 베이스를 활용하는 RAG 챗봇"""
    
    def __init__(self, agent: FDSEnableAgent, context_builder: FDSContextBuilder):
        """RAG 챗봇을 초기화한다."""
        self.agent = agent
        self.context_builder = context_builder
        self.client = OpenAI()
        self.conversation_history: List[Dict[str, Any]] = []
        
        print("FDS RAG 챗봇 초기화 완료")
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트를 생성한다."""
        knowledge = self.context_builder.get_knowledge_base_content()
        summary = self.context_builder.get_summary()
        
        return f"""
당신은 사기 탐지 시스템(FDS) AI 어시스턴트다.

## 역할

1. 거래 정보를 받아 사기 가능성을 분석한다
2. 과거 분석 기록을 바탕으로 통계와 인사이트를 제공한다
3. 분석 결과를 친절하게 설명한다
4. 위험도에 따른 대응 조치를 권장한다

## 현재 통계

{json.dumps(summary, indent=2, ensure_ascii=False) if summary else '분석 데이터 없음'}

## 지식 베이스

{knowledge if knowledge else '분석 데이터 없음'}

## 위험도 등급 및 권장 조치

- LOW (0~0.3): 정상 처리
- MEDIUM (0.3~0.5): 모니터링 강화
- HIGH (0.5~0.7): 추가 인증 요청
- CRITICAL (0.7~1.0): 거래 보류 및 수동 검토

## 응답 규칙

- 문장은 ~다로 끝낸다
- 거래 분석 요청 시 analyze_transaction 함수를 호출한다
- 전문적이지만 친절하게 설명한다
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
                
                if function_name == "analyze_transaction":
                    result = self.agent.analyze(function_args)
                    self.context_builder.add_analysis(result)
                    
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
        knowledge = self.context_builder.get_knowledge_base_content()
        statistics = self.context_builder.get_statistics()
        
        messages = [
            {
                "role": "system",
                "content": "당신은 FDS 분석 보고서를 작성하는 전문가다. 문장은 ~다로 끝낸다."
            },
            {
                "role": "user",
                "content": f"""
다음 FDS 분석 데이터를 바탕으로 상세 보고서를 작성해달라:

## 요약 통계
{json.dumps(summary, indent=2, ensure_ascii=False)}

## 상세 통계
{json.dumps(statistics, indent=2, ensure_ascii=False)}

## 분석 기록
{knowledge}

보고서 내용:
1. 전체 분석 현황 요약
2. 사기 탐지 패턴 분석
3. 위험 요인 분석
4. 권장 조치 사항
5. 시스템 개선 방향
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


def interactive_chat(agent: FDSEnableAgent, context_builder: FDSContextBuilder):
    """인터랙티브 챗봇 세션을 시작한다."""
    chatbot = FDSRAGChatbot(agent, context_builder)
    
    print("\n" + "=" * 60)
    print("FDS 사기 탐지 RAG 챗봇")
    print("=" * 60)
    print("종료: 'quit' 또는 'exit'")
    print("보고서: 'report'")
    print("초기화: 'reset'")
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
            print("FDS 분석 보고서")
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
    from .enable_agent import FDSEnableAgent
    from .context_builder import FDSContextBuilder
    
    agent = FDSEnableAgent('skills/agent_skill.yaml')
    context_builder = FDSContextBuilder()
    
    interactive_chat(agent, context_builder)
