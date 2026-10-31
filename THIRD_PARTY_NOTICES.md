# Third-party notices

Original PathFM material is licensed under [CC BY-NC 4.0](LICENSE).
Third-party material retains its original licenses and notices as described below.
The package license expression combines the project license with the Apache-2.0
and MIT licenses represented in the bundled third-party material.

## Code

The model uses the Chronos-2 implementation from `chronos-forecasting`, distributed under Apache License 2.0.

FEV uses the published `fev==0.9.0` package from `autogluon/fev`.
Its license is retained in `src/pathfm/benchmarks/protocols/FEV-LICENSE`.

`gift_data.py` and `gift-dataset-properties.json` derive from `SalesforceAIResearch/gift-eval`.
The source copyright and Apache License 2.0 notice are retained in the loader.

`time_data.py` and `time-datasets.yaml` derive from `zqiao11/TIME`.
The TIME data loader retains its original Salesforce Apache-2 source notice and TIME modification notice.

The PANDA window-sampling and sMAPE conventions follow `abao1999/panda`, distributed under the MIT License.
Its license is retained in `src/pathfm/benchmarks/protocols/PANDA-LICENSE`.

## External datasets and model checkpoints

Benchmark datasets downloaded from Hugging Face and model checkpoints loaded by
`from_pretrained` are governed by the licenses and terms provided with their source
repositories. The PathFM project license does not replace those licenses or terms.
