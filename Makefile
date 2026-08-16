.PHONY: demo test demo-claude clean

# Основная команда: полный прогон всех 6 сценариев. Без сети и без ключей.
demo:
	python3 poc/run_demo.py

# Тот же прогон, но черновики генерирует реальный Claude API.
# Требует: export ANTHROPIC_API_KEY=...
demo-claude:
	python3 poc/run_demo.py --provider claude

test:
	python3 -m unittest discover -s tests -v

clean:
	rm -f logs/decisions.jsonl
	find . -name "__pycache__" -type d -exec rm -rf {} +
