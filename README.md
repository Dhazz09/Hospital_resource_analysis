#  Predictive Analysis for Hospital Resource Optimisation

##  Overview:
This project provides a real-time dashboard and API to forecast bed usage, oxygen demand, and staff allocation in hospitals using machine learning.

---

## 📂 Components
- **Backend (FastAPI)**: REST API for ingesting hospital data and generating forecasts.
- **Trainer**: Simple ML pipeline (Linear Regression baseline).
- **Dashboard (Streamlit)**: Interactive UI for visualising trends and predictions.
- **Docker Compose**: Orchestrates backend + dashboard.

---

## 🛠️ Setup
Clone repo and run:

```bash
docker-compose up --build
