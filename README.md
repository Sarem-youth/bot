# Gold Trading Bot - Development Setup

## Development Environment

### Prerequisites
- Python 3.8+
- Git
- Visual Studio Code or PyCharm Professional
- Docker (optional)

### IDE Setup
#### VSCode Extensions
- Python
- Pylance
- Python Test Explorer
- GitLens
- Docker
- Error Lens

#### PyCharm Plugins
- MetaProgramming Assistant
- Git Toolbox
- Requirements

### Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Code Quality Tools
- flake8: Style guide enforcement
- mypy: Static type checking
- black: Code formatting
- isort: Import sorting
- pylint: Code analysis

### Testing Framework
- pytest: Unit and integration testing
- pytest-cov: Code coverage
- pytest-mock: Mocking functionality
- pytest-asyncio: Async test support

### Git Workflow
1. Create feature branch
2. Write tests
3. Implement feature
4. Run tests & quality checks
5. Create pull request

### CI/CD Pipeline
- GitHub Actions for automation
- Pre-commit hooks for code quality
- Automated testing on push
- Docker image building

### Monitoring & Debugging
- logging configuration
- debugpy for remote debugging
- VSCode debug configurations

## Cloud Infrastructure Setup

### AWS Environment
```bash
# Set up AWS infrastructure
aws configure
terraform init
terraform apply

# Deploy trading bot to EKS
kubectl apply -f deployment.yaml
```

### Azure Setup
```bash
# Configure Azure environment
az login
az aks get-credentials --resource-group trading --name trading-cluster

# Set up Azure Container Registry
az acr build -t gold-trading-bot:latest .
```

### Monitoring Setup
```bash
# Install monitoring stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack

# Configure Grafana dashboards
kubectl apply -f monitoring/
```

## Collaboration Guidelines

### Code Review Process
1. Create feature branch from develop
2. Submit PR with required reviewers
3. Pass automated checks
4. Obtain approvals
5. Merge using squash

### Documentation
- API changes require OpenAPI spec updates
- Architecture changes need design doc
- Update runbooks for operational changes
- Maintain change log

### Security
- Secrets in HashiCorp Vault
- Rotate keys every 90 days
- Regular security scans
- Access audit quarterly

### Release Process
1. Version bump
2. Update CHANGELOG.md
3. Create release branch
4. Deploy to staging
5. Run integration tests
6. Deploy to production

## Environment Variables
```bash
# Required environment variables
MT5_SERVER=broker.example.com
MT5_ACCOUNT=12345
MT5_PASSWORD=secret
AZURE_CONNECTION_STRING=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```
