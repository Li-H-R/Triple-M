Overview
This project was originally built based on the Transformer architecture, which has demonstrated strong performance in sequence modeling and attention-based tasks.
Update: LSTM Integration
In the latest version, we have extended the model by incorporating a Long Short-Term Memory (LSTM) architecture alongside the existing Transformer-based framework.
Motivation
While Transformer models excel at capturing global dependencies through attention mechanisms, LSTM networks are effective at modeling sequential patterns and temporal dependencies. By introducing LSTM, we aim to:
Enhance sequence modeling capabilities
Improve performance on tasks with strong temporal structure
Provide architectural flexibility for experimentation
Architecture
The current implementation supports:
Transformer-based model (original baseline)
LSTM-based model (newly added)
Configurable selection between architectures
