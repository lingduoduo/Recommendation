#!/usr/bin/env python
import click

from util.load_model import predict_ranking_model


# To run command line test
# ./cli_ranking.py --jsoninput "kik:user:unknown" "kik:user:unknown" "Rose" "10" "train_dcn_ranking"


@click.command()
@click.option(
    "--jsoninput",
    nargs=5,
    type=(str, str, str, str, str),
    default=("kik:user:unknown", "kik:user:unknown", "Rose", "10", "train_dcn_ranking"),
)
def predictcli(jsoninput):
    """Predicts recommendations"""
    user, broadcaster, product_name, order_time, path = jsoninput
    score = predict_ranking_model(user, broadcaster, product_name, order_time, path)
    print(
        f"--- recommendations for viewer={user} to broadcaster={broadcaster} with previously received gift_name={product_name} time_slot={order_time} --- using the {path} model "
    )
    readable = f"{broadcaster}: {score}"
    if float(score) > 0.02:
        click.echo(click.style(readable, bg="green", fg="white"))
    else:
        click.echo(click.style(readable, bg="red", fg="white"))


if __name__ == "__main__":
    predictcli()
