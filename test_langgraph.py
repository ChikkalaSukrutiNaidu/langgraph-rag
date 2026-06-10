from services.graph_builder import app

result = app.invoke(
    {
        "question": "What is this document about?"
    }
)

print(result["answer"])