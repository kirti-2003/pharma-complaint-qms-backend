from app.ai.clients.groq_client import groq_client


def test_groq_connection():
    result = groq_client.generate_completion(
        system_prompt=(
            "You are a pharmaceutical complaint assistant."
        ),
        user_prompt=(
            "Reply with only this text: Groq connection successful"
        ),
        temperature=0,
        max_tokens=50,
    )

    print(result["content"])
    print("Model:", result["model"])
    print("Total tokens:", result["total_tokens"])


if __name__ == "__main__":
    test_groq_connection()