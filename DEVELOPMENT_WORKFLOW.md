# Development Workflow Guide

## Branch Structure

Your project now has a professional three-branch workflow:

### 🚀 **main** (Production Branch)
- **Purpose**: Production-ready, stable code
- **Protection**: Direct pushes should be restricted
- **Deployment**: Automatically deploys to production
- **Status**: ✅ Current stable version

### 🧪 **testing** (Staging Branch)  
- **Purpose**: Integration testing and staging
- **Source**: Merges from `development` branch
- **Testing**: Automated tests run via GitHub Actions
- **Status**: ✅ Ready for testing

### 💻 **development** (Development Branch)
- **Purpose**: Active development work
- **Source**: Feature branches merge here first
- **Status**: ✅ Ready for development

## Development Process

### 1. Feature Development
```bash
# Start from development branch
git checkout development
git pull origin development

# Create feature branch
git checkout -b feature/your-feature-name

# Develop and commit
git add .
git commit -m "feat: Add your feature description"
git push origin feature/your-feature-name
```

### 2. Code Review Process
```bash
# Create Pull Request: feature/branch → development
# After review and approval, merge to development
git checkout development
git pull origin development
```

### 3. Testing & Staging
```bash
# Merge development to testing for integration testing
git checkout testing
git pull origin testing
git merge development
git push origin testing
```

### 4. Production Release
```bash
# After successful testing, merge to main
git checkout main
git pull origin main
git merge testing
git push origin main
```

## Environment Setup

### Development Environment
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set environment
export FLASK_ENV=development
export FLASK_DEBUG=1

# Run with development config
python app.py
```

### Testing Environment
```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Set testing environment
export FLASK_ENV=testing
```

### Production Environment
```bash
# Install only production dependencies
pip install -r requirements.txt

# Set production environment
export FLASK_ENV=production
export FLASK_DEBUG=0
```

## Configuration Management

The project uses environment-specific configurations in `config.py`:

- **DevelopmentConfig**: Debug enabled, detailed logging
- **TestingConfig**: Testing mode, isolated environment  
- **ProductionConfig**: Optimized for performance and security

## CI/CD Pipeline

GitHub Actions automatically:
- ✅ Runs tests on pull requests
- ✅ Checks code quality (linting)
- ✅ Validates Python syntax
- ✅ Tests multiple Python versions (3.8, 3.9, 3.10)

## Quick Commands

```bash
# Check current branch
git branch

# Switch branches
git checkout main
git checkout development  
git checkout testing

# View branch status
git status

# See recent commits
git log --oneline -5

# Run development server
python app.py

# Run tests
pytest

# Code formatting
black .

# Linting
flake8 .
```

## Team Collaboration

1. **Always** start new features from `development` branch
2. **Never** push directly to `main` or `testing`
3. **Always** create pull requests for code review
4. **Test** your changes before creating pull requests
5. **Follow** the commit message format: `type: description`

## Commit Message Types
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `style:` Code style changes

---

🎉 **Your project is now ready for professional collaborative development!**