import streamlit as st

def main():
    st.title("AI 채팅봇")

    # 세션 상태에 메시지 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # 사용자 입력
    user_input = st.chat_input("메시지를 입력하세요...")

    if user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AI 응답 (여기서는 간단한 에코 응답)
        ai_response = f"AI: {user_input}"
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # 채팅 컨테이너 업데이트
        with chat_container:
            for message in st.session_state.messages[-2:]:  # 마지막 두 메시지만 표시
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    # 스크롤을 최신 메시지로 이동
    if st.session_state.messages:
        st.rerun()

if __name__ == "__main__":
    main()