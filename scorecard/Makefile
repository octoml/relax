# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

TEST_DATA ?= build/testdata.jsonl

clean:
	rm -rf build

build/venv-created.touch: poetry.lock pyproject.toml
	poetry install
	mkdir -p build
	touch build/venv-created.touch

prepare_testdata: build/venv-created.touch testdata/*.jsonc schema/schema.jsonschema relax_scorecard/*.py
	mkdir -p build
	poetry run python3 -m relax_scorecard.concat_testdata --schema schema/schema.jsonschema testdata/*.jsonc >"${TEST_DATA}"
	@echo "Prepared testdata in ${TEST_DATA}"


.DEFAULT_GOAL = prepare_testdata
