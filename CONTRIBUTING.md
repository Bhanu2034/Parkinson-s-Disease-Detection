# Contributing to ParkinsonAI

We welcome contributions from the community! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and professional in all interactions. We expect all contributors to:
- Be respectful of differing opinions and approaches
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of Flask, scikit-learn, and audio processing

### Setup for Development

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/parkinsons-hybrid-detection.git
   cd Final_Parkinson_Proj-main
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Making Changes

### Code Style
- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and modular

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (e.g., "Add", "Fix", "Update", "Refactor")
- Example: `Add voice feature normalization in audio processing`

### Testing
- Test your changes locally before submitting
- Run the app and verify all features work correctly
- Test with different input data to ensure robustness

## Submitting Changes

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - Clear title describing the changes
   - Detailed description of what was changed and why
   - Reference to any related issues
   - Screenshots if UI changes were made

3. **Wait for review** - A maintainer will review your PR and provide feedback

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Enhancement
- [ ] Documentation update

## Related Issues
Closes #(issue number)

## Testing
Describe how you tested these changes

## Checklist
- [ ] Code follows PEP 8 style guidelines
- [ ] I have performed a self-review
- [ ] Comments added for complex code
- [ ] No new warnings generated
- [ ] Changes tested locally
```

## Areas for Contribution

### High Priority
- [ ] Improve model accuracy with new architectures
- [ ] Add support for different audio formats
- [ ] Enhance UI/UX
- [ ] Add unit tests
- [ ] Documentation improvements

### Medium Priority
- [ ] Add data augmentation techniques
- [ ] Implement additional evaluation metrics
- [ ] Create notebook examples
- [ ] Add more voice biomarkers analysis

### Lower Priority
- [ ] Code optimization
- [ ] Additional visualizations
- [ ] Database integration
- [ ] API improvements

## Questions or Need Help?

- **Found a bug?** Open an issue with:
  - Clear description of the problem
  - Steps to reproduce
  - Expected vs actual behavior
  - System information (OS, Python version)

- **Have a feature suggestion?** Create a discussion or issue with:
  - Use case description
  - Expected behavior
  - Any relevant references

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Project acknowledgments
- Release notes

Thank you for contributing to ParkinsonAI!
