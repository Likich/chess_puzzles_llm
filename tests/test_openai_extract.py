from chess_reasoning.generation.openai_api import extract_output_text


def test_extract_output_text_from_output():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "move"},
                    {"type": "output_text", "text": "explain"},
                ],
            }
        ]
    }
    text = extract_output_text(response)
    assert text == "move\nexplain"


def test_extract_output_text_from_top_level():
    response = {"output_text": "hello"}
    assert extract_output_text(response) == "hello"
