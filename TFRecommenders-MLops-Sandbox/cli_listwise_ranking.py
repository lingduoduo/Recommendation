#!/usr/bin/env python
import click

from util.load_model import predict_listwise_ranking_model


# To run command line test
# ./cli_listwise_ranking.py --jsoninput "kik:user:unknown" "train_listwise_ranking"


@click.command()
@click.option(
    "--jsoninput",
    nargs=2,
    type=(str, str),
    default=("kik:user:unknown", "train_listwise_ranking"),
)
def predictcli(jsoninput):
    """Predicts recommendations"""
    user, path = jsoninput
    pred = predict_listwise_ranking_model(user, path)
    print(f"--- recommendations for {user} --- using the {path} model ")
    for broadcaster, score in sorted(pred, key=lambda x: x[1], reverse=True):
        readable = f"{broadcaster}: {score}"
        if float(score) > 0.02:
            click.echo(click.style(readable, bg="green", fg="white"))
        else:
            click.echo(click.style(readable, bg="red", fg="white"))


if __name__ == "__main__":
    predictcli()
