if groq_api_key and user_query:

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
    )

    if use_doc_context and st.session_state.vector_store:

        from rag_module import advanced_retrieval

        context_chunks = advanced_retrieval(
            user_query,
            st.session_state.vector_store,
            st.session_state.doc_chunks
        )

        st.subheader("🔎 Retrieved Context (Multi-Query)")
        for i, chunk in enumerate(context_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk[:300]}...")

        reply = answer_query_with_context(user_query, context_chunks)

    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_query}],
            temperature=0.7,
        )
        reply = response.choices[0].message.content

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
