# UML 設計文書

この文書は現在の実装を Mermaid UML で説明します。Markdown viewer が Mermaid に対応していれば図として表示されます。

## Component 図

```mermaid
flowchart LR
    CLI[trt-profiler CLI] --> Loader[config.loader]
    Loader --> Parser[config.parser]
    Parser --> Schema[config.schema]
    Loader --> Validation[config.validation]
    Loader --> ConfigBuilder[config.builder]
    ConfigBuilder --> Presets[config.presets]
    CLI --> Pipeline[EvaluationPipeline]
    Pipeline --> Factory[core.factory]
    Factory --> Builders[builders]
    Factory --> Runners[runners]
    Factory --> Data[data loaders]
    Factory --> Preprocessors[preprocessors]
    Factory --> Postprocessors[postprocessors]
    Factory --> Metrics[metrics]
    Pipeline --> Mapping[input/output mapping]
    Pipeline --> Evaluators[evaluators]
    Pipeline --> ReportData[report.data_builder]
    ReportData --> Reporters[reporters]
    Reporters --> Json[report.json]
    Reporters --> Csv[csv files]
    Reporters --> Html[dashboard.html]
    Json --> Dash[Dash server]
```

## Config load class 図

```mermaid
classDiagram
    class EvaluationConfig {
        +ModelConfig model
        +InputConfig input
        +dict outputs
        +list~VariantConfig~ variants
        +MetricsConfig metrics
        +ComponentConfig preprocess
        +list~ComponentConfig~ postprocessors
        +ReportConfig report
        +str compare
        +dict tensor_rt
    }

    class ModelConfig {
        +str name
        +str source_path
        +str format
        +list~str~ tasks
    }

    class InputConfig {
        +str kind
        +str path
        +str input_name
        +str backend_name
        +str pattern
        +str npz_key
        +str dtype
    }

    class VariantConfig {
        +str name
        +str preset
        +str role
        +dict builder
        +dict runner
        +dict extra
    }

    class ComponentConfig {
        +str name
        +str preset
        +str class_path
        +dict config
    }

    class MetricConfig {
        +str name
        +str preset
        +str class_path
        +dict config
    }

    class MetricsConfig {
        +list~MetricConfig~ raw
        +list~MetricConfig~ post
    }

    class ReportConfig {
        +str output_dir
        +list~str~ formats
    }

    EvaluationConfig --> ModelConfig
    EvaluationConfig --> InputConfig
    EvaluationConfig --> VariantConfig
    EvaluationConfig --> MetricsConfig
    EvaluationConfig --> ComponentConfig
    EvaluationConfig --> ReportConfig
    MetricsConfig --> MetricConfig
```

## Core contract class 図

```mermaid
classDiagram
    class DatasetLoader {
        <<abstract>>
        +dict config
        +__iter__() Iterator~Sample~
    }

    class Preprocessor {
        <<abstract>>
        +dict config
        +validate_config() void
        +__call__(Sample) TensorDict
    }

    class Postprocessor {
        <<abstract>>
        +dict config
        +validate_config() void
        +__call__(TensorDict, Sample) ComparableDict
    }

    class ArtifactBuilder {
        <<abstract>>
        +str name
        +str backend
        +str precision
        +dict config
        +validate_config() void
        +build(SourceModel) ModelArtifact
    }

    class ModelRunner {
        <<abstract>>
        +str name
        +ModelArtifact artifact
        +dict config
        +validate_config() void
        +load() void
        +infer(TensorDict) TensorDict
        +get_input_specs() list~TensorSpec~
        +get_output_specs() list~TensorSpec~
        +warmup(TensorDict, int) void
    }

    class Metric {
        <<abstract>>
        +str name
        +dict config
        +validate_config() void
        +update(dict, dict, Sample) list~SampleMetricRecord~
        +compute() list~MetricSummaryRecord~
    }

    class Reporter {
        <<abstract>>
        +dict config
        +write(ReportData) void
    }

    class EvaluationPipeline {
        +dict config
        +run() EvaluationResult
    }

    EvaluationPipeline --> DatasetLoader
    EvaluationPipeline --> Preprocessor
    EvaluationPipeline --> Postprocessor
    EvaluationPipeline --> ArtifactBuilder
    EvaluationPipeline --> ModelRunner
    EvaluationPipeline --> Metric
    EvaluationPipeline --> Reporter
```

## Data class 図

```mermaid
classDiagram
    class SourceModel {
        +Path path
        +str format
    }

    class ModelArtifact {
        +str variant_name
        +str backend
        +str precision
        +Path path
        +dict config
    }

    class BackendVariant {
        +str name
        +str backend
        +str role
        +str builder_class
        +str runner_class
        +dict builder_config
        +dict runner_config
        +str precision
    }

    class Comparison {
        +str name
        +str reference
        +str target
    }

    class Sample {
        +str id
        +Any data
        +Path source_path
        +Any label
        +list annotations
        +dict metadata
    }

    class SampleMetricRecord {
        +str sample_id
        +str comparison
        +str stage
        +str metric
        +str stat
        +Any value
        +str output
        +float threshold
        +str status
        +str source_path
    }

    class MetricSummaryRecord {
        +str comparison
        +str stage
        +str metric
        +str stat
        +Any value
        +str output
        +float threshold
        +str status
    }

    class EvaluationResult {
        +dict metadata
        +dict summary
        +list~SampleMetricRecord~ per_sample
    }

    class ReportData {
        +dict metadata
        +dict summary
        +dict tables
        +dict artifacts
    }

    BackendVariant --> ModelArtifact
    Comparison --> BackendVariant
    EvaluationResult --> SampleMetricRecord
    EvaluationResult --> MetricSummaryRecord
    ReportData --> EvaluationResult
```

## 評価 sequence 図

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Loader as config.loader
    participant Pipeline as EvaluationPipeline
    participant Builder as ArtifactBuilder
    participant Runner as ModelRunner
    participant Dataset as DatasetLoader
    participant Pre as Preprocessor
    participant Metric as Metric
    participant Post as Postprocessor
    participant Reporter

    User->>CLI: trt-profiler eval -c config.yaml
    CLI->>Loader: load_config(config.yaml)
    Loader-->>CLI: full pipeline config
    CLI->>Pipeline: EvaluationPipeline(config).run()
    Pipeline->>Builder: build(SourceModel)
    Builder-->>Pipeline: ModelArtifact
    Pipeline->>Runner: load()
    loop sample
        Pipeline->>Dataset: next Sample
        Pipeline->>Pre: __call__(Sample)
        Pre-->>Pipeline: common TensorDict
        Pipeline->>Runner: infer(mapped TensorDict)
        Runner-->>Pipeline: backend TensorDict
        Pipeline->>Metric: raw update(reference, target, sample)
        Pipeline->>Post: __call__(outputs, sample)
        Post-->>Pipeline: ComparableDict
        Pipeline->>Metric: post update(reference, target, sample)
    end
    Pipeline->>Metric: compute()
    Metric-->>Pipeline: summary records
    Pipeline->>Reporter: write(ReportData)
```

## Config load sequence 図

```mermaid
sequenceDiagram
    participant Loader as load_config
    participant Parser as parse_evaluation_config
    participant Validator as validate_evaluation_config
    participant Builder as build_pipeline_config
    participant Presets as presets

    Loader->>Loader: yaml.safe_load
    alt common exists
        Loader-->>Loader: return full config
    else concise config
        Loader->>Parser: raw dict
        Parser-->>Loader: EvaluationConfig
        Loader->>Validator: EvaluationConfig
        Loader->>Builder: EvaluationConfig
        Builder->>Presets: resolve variants / metrics / components / reporters
        Presets-->>Builder: full component configs
        Builder-->>Loader: full pipeline config
    end
```

## 拡張時の依存方向

```mermaid
flowchart TD
    CustomPre[Custom Preprocessor] --> CoreTypes[core.types.Preprocessor]
    CustomPost[Custom Postprocessor] --> CoreTypes2[core.types.Postprocessor]
    CustomMetric[Custom Metric] --> CoreTypes3[core.types.Metric]
    CustomRunner[Custom Runner] --> CoreTypes4[core.types.ModelRunner]
    CustomBuilder[Custom Builder] --> CoreTypes5[core.types.ArtifactBuilder]
    Pipeline[EvaluationPipeline] --> CoreTypes
    Pipeline --> CoreTypes2
    Pipeline --> CoreTypes3
    Pipeline --> CoreTypes4
    Pipeline --> CoreTypes5
    Pipeline -. does not know .-> TaskSpecific[task-specific schema]
```

評価 core はタスク固有 schema に依存しません。独自処理は基底 class の契約を満たし、config の dotted path で指定します。

