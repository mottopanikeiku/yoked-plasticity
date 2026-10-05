"""Summarize real simulator, replay, and neural work without double-counting data."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    config = json.loads((directory / "config.json").read_text())
    records = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    totals = dict(training_simulator_interactions=0, reused_consumer_transition_exposures=0,
                  evaluation_simulator_interactions=0, exact_reference_transition_queries=0,
                  neural_optimizer_updates=0, tabular_minibatch_updates=0,
                  instrumented_network_forward_examples=0, frozen_network_forward_examples=0,
                  evaluation_network_forward_examples=0)

    def add_work(work):
        totals["instrumented_network_forward_examples"] += work["network_forward_examples"]
        totals["frozen_network_forward_examples"] += work["frozen_network_forward_examples"]
        totals["evaluation_network_forward_examples"] += work["evaluation_network_forward_examples"]

    for record in records:
        totals["training_simulator_interactions"] += record["age"]["training_interactions"] + record["tabular_age"]["training_interactions"]
        totals["neural_optimizer_updates"] += record["age"]["optimizer_updates"]
        totals["tabular_minibatch_updates"] += record["tabular_age"]["optimizer_updates"]
        add_work(record["age"]["work"])
        for switch in record["switches"].values():
            totals["training_simulator_interactions"] += switch["training_simulator_interactions"]
            totals["evaluation_simulator_interactions"] += switch["actor_evaluation_interactions"]
            totals["exact_reference_transition_queries"] += switch["reference_transition_queries"]
            totals["neural_optimizer_updates"] += switch["actor_optimizer_updates"]
            for work in switch["actor_work"].values():
                add_work(work)
            consumers = list(switch["cells"].values()) + list(switch["clean_tape_calibration"].values())
            totals["reused_consumer_transition_exposures"] += len(consumers) * config["adapt_steps"]
            for consumer in consumers:
                totals["neural_optimizer_updates"] += consumer["optimizer_updates"]
                totals["evaluation_simulator_interactions"] += consumer["evaluation_interactions"]
                add_work(consumer["work"])
            for control in switch["controls"].values():
                totals["neural_optimizer_updates"] += control.get("optimizer_updates", 0)
                totals["tabular_minibatch_updates"] += control.get("q_minibatch_updates", 0)
                totals["evaluation_simulator_interactions"] += control["evaluation_interactions"]
                add_work(control["work"])
    totals["neural_sampled_replay_items"] = totals["neural_optimizer_updates"] * config["batch_size"]
    totals["tabular_sampled_replay_items"] = totals["tabular_minibatch_updates"] * config["batch_size"]
    # run_pair checks all ten observations once for each of A and I, outside
    # phase counters. Injection evaluates 1+3 networks; identity evaluates 1+1.
    totals["derived_initial_equivalence_check_forward_examples"] = len(records) * {"injection": 40, "identity": 20, "head_reset": 0}[config["intervention"]]
    totals["all_training_transition_exposures"] = totals["training_simulator_interactions"] + totals["reused_consumer_transition_exposures"]
    return {"run": directory.name, "pair_count": len(records), "counts": totals,
            "notes": "A network-forward example counts one observation evaluated by one MLP, not a FLOP estimate. Initial equivalence-check work is derived separately from the frozen runner; all other forward counts are instrumented. Tabular updates are not neural optimizer steps. Consumer reuse is not new simulator experience."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(summarize(arguments.run), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")
