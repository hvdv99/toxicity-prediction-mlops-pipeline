# Toxic Predictor

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
      <ul>
        <li><a href="#ci_cd_pipelines">CI/CD Pipelines</a></li>
        <li><a href="#components">Components</a></li>
        <li><a href="#vertex_ai_pipeline">Vertex AI Pipeline</a></li>
      </ul>
    <li><a href="#GCP Architecture">CP Architecture</a></li>
    <li><a href="#authors">Authors</a></li>
  </ol>
</details>

## About The Project
In today's digital era, ensuring our messages don't unintentionally harm or offend is crucial.  
Our app predicts the toxicity of your messages across six categories:  
```[Toxic, Severe toxic, Obscene, Insult, Identity hate, Threat]```  
Using a multi-label chaining classifier, it assesses each message and scores them for potential harm.Built with modularity in mind, each model component is containerized using Docker. Our CI/CD process guarantees up-to-date models, automatically retraining and deploying them when code or data changes.  
Ensure your messages are safe before you hit _send_!
### Built With
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB)](https://www.python.org/downloads/)
## Project Structure
```angular2html
├───ci_cd_pipelines
│   ├───components_builder
│   ├───ml_pipeline_executor
│   └───serving_deployment
├───components
│   ├───toxic-data-cleaner
│   ├───toxic-data-ingestor
│   ├───toxic-multilabel-trainer
│   ├───toxic-prediction-ui
│   ├───toxic-predictor
│   └───toxic-train-test-split
└───vertex_ai_pipeline
```
### CI/CD Pipelines
- **Component Builder:** Automates the building of Docker images from each component's source code and pushes them to the Google Cloud Image Registry.
- **ML Pipeline Executor:** Orchestrates the execution of the ML pipeline within Vertex AI using individual components.
- **Serving Component Deployer:** Deploys three APIs - Cleaning, Prediction, and UI. Uses URL from previous steps to interlink the services.  
_Triggering Mechanism_: A push within the components folder of the GitHub repo triggers the Component Builder. This, in turn, initiates the ML Pipeline Executor, which then triggers the Deployment Pipeline.

### ML Components
- **toxic-data-ingestor**: Fetches data from Google Cloud Storage Databucket.
- **toxic-data-cleaner**: Processes and normalizes raw data for better model outcomes.
- **toxic-train-test-split**: Divides the cleaned dataset into training and test samples.
- **toxic-multilabel-trainer**: Consumes the cleaned data, vectorizes text, trains models for each toxicity category, and integrates the new models into the prediction API.
- **toxic-predictor**: Employs the trained models to determine toxicity labels for the test dataset.

### Vertex AI Pipeline
Contains a notebook that is used to generate a ```.yaml``` pipeline using Kubeflow components.  
This pipeline leverages the individual ML components described above to create a seamless and automated ML workflow.

## GCP Architecture
![image](https://github.com/RomanNekrasov/DataEngineering/assets/33453661/247c940f-e435-4d91-88c6-5d2031c7b7d5)

## Authors
- Andy Huang
- Huub van de Voort 
- Oumaima Lemhour
- Roman Nekrasov
- Tom Teurlings
