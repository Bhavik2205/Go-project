run:
	docker build -t ml-bot .
	docker run --rm -it --name ml-bot ml-bot

shell:
	docker run --rm -it -v $(PWD):/app --name ml-bot-dev ml-bot bash
