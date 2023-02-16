import argparse
import commentjson
import json
import jsonschema
import pathlib
import re
import sys


def _load_and_strip_comments(f):
    return commentjson.loads(f.read())


def concat_test_results(schema, test_results, output):
    for path in test_results:
        with open(path, "r") as f:
            try:
                data = _load_and_strip_comments(f)
            except Exception as e:
                print(f"while loading {path}:", file=sys.stderr)
                raise e

        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.SchemaError as e:
            print(f"while validating {path}:", file=sys.stderr)
            raise e
        output.write(json.dumps({"r": json.dumps(data)}))
        output.write("\n")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema",
        type=pathlib.Path,
        required=True,
        help="Path to the JSON schema describing the test data",
    )
    parser.add_argument(
        "test_results",
        type=pathlib.Path,
        nargs="+",
        help="Path to test results which should be concatenated",
    )

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    with open(args.schema) as f:
        schema = _load_and_strip_comments(f)

    concat_test_results(schema, args.test_results, sys.stdout)
    print(
        f"Prepared {len(args.test_results)} test results for upload to BigQuery",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
