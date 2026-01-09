# Medical GraphRAG with Qwen Integration

> **Medical domain adaptation of Microsoft GraphRAG with integrated Qwen 3-30B chat and Qwen 3 Embedding 0.6B models for healthcare knowledge graphs**

👉 [Original Microsoft GraphRAG](https://github.com/microsoft/graphrag)<br/>
👉 [Microsoft Research Blog Post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)<br/>
👉 [GraphRAG Documentation](https://microsoft.github.io/graphrag)<br/>
👉 [GraphRAG Arxiv Paper](https://arxiv.org/pdf/2404.16130)

<div align="left">
  <a href="https://github.com/myhearwillgoon/Medical_graphrag">
    <img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Medical%20GraphRAG-blue">
  </a>
  <a href="https://github.com/myhearwillgoon/Medical_graphrag/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/myhearwillgoon/Medical_graphrag">
  </a>
  <a href="https://github.com/myhearwillgoon/Medical_graphrag/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  </a>
</div>

## 🎯 Overview

This repository is a **medical domain adaptation** of [Microsoft GraphRAG](https://github.com/microsoft/graphrag) with integrated **Qwen model support** for healthcare knowledge graph applications. It extends the original GraphRAG framework with:

- **Qwen 3-30B Chat Model** integration via OpenAI-compatible API
- **Qwen 3 Embedding 0.6B Model** for efficient vector embeddings
- Medical domain optimizations and configurations
- Enhanced language model factory supporting custom endpoints

## ✨ Key Features

### Qwen Model Integration
- **Qwen Chat Provider** (`qwen_chat.py`) - Full OpenAI-compatible API implementation
- **Qwen Embedding Provider** (`qwen_embedding.py`) - Batch embedding support with efficient processing
- Seamless integration with existing GraphRAG architecture
- Configurable endpoint support for self-hosted Qwen models

### Medical Domain Focus
- Optimized for healthcare knowledge graph construction
- Medical terminology and domain-specific prompt templates
- Enhanced entity extraction for medical concepts
- Community detection for medical knowledge clusters

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Qwen model servers running:
  - Chat API: `http://your-server:port/v1/chat/completions`
  - Embedding API: `http://your-server:port/v1/embeddings`

### Installation

```bash
# Clone the repository
git clone https://github.com/myhearwillgoon/Medical_graphrag.git
cd Medical_graphrag

# Install in development mode
pip install -e .
```

### Configuration

Configure Qwen endpoints in your `settings.yaml`:

```yaml
language_model:
  chat:
    provider: qwen
    model: "your-qwen-chat-model"
    api_base: "http://your-server:port/v1/chat/completions"
  
  embedding:
    provider: qwen
    model: "your-qwen-embedding-model"
    api_base: "http://your-server:port/v1/embeddings"
```

## 📋 What's Different from Original GraphRAG?

### Core Modifications
1. **New Providers**: `qwen_chat.py` and `qwen_embedding.py` in `graphrag/language_model/providers/`
2. **Factory Updates**: Enhanced language model factory to support Qwen endpoints
3. **Configuration**: Extended defaults and enums for Qwen model support
4. **FNL LM Enhancements**: Improved compatibility with Qwen-specific configurations

### Files Added
- `graphrag/language_model/providers/qwen_chat.py` - Qwen chat implementation
- `graphrag/language_model/providers/qwen_embedding.py` - Qwen embedding implementation
- `.spec-workflow/` - Comprehensive specification documentation

### Files Modified
- `graphrag/config/defaults.py` - Qwen configuration defaults
- `graphrag/config/enums.py` - Qwen enum additions
- `graphrag/language_model/factory.py` - Factory pattern updates
- `graphrag/language_model/providers/fnllm/models.py` - Enhanced FNL LM support

## 📚 Documentation

- **Spec Workflow**: See `.spec-workflow/specs/` for detailed requirements, design, and task documentation
- **Original GraphRAG Docs**: [Microsoft GraphRAG Documentation](https://microsoft.github.io/graphrag)
- **Qwen Integration**: Check provider implementations in `graphrag/language_model/providers/`

## ⚠️ Important Notes

- **Based on**: Microsoft GraphRAG v2.7.0
- **License**: MIT (inherited from original GraphRAG project)
- **Not Officially Supported**: This is a community adaptation, not an official Microsoft offering
- **Medical Use**: Ensure compliance with healthcare data regulations (HIPAA, GDPR, etc.)

## 🤝 Contributing

This repository is based on Microsoft GraphRAG. For contributions:
- Report issues specific to this Medical GraphRAG adaptation
- For original GraphRAG issues, see [Microsoft GraphRAG Issues](https://github.com/microsoft/graphrag/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright Notice**: This is a derivative work based on Microsoft GraphRAG. Original copyright belongs to Microsoft Corporation.

## 🙏 Acknowledgments

- **Microsoft GraphRAG Team** - For the excellent original GraphRAG framework
- **Qwen Team** - For the powerful Qwen language models
- **Open Source Community** - For continuous improvements and feedback

## 🔗 Related Links

- [Original Microsoft GraphRAG Repository](https://github.com/microsoft/graphrag)
- [Microsoft GraphRAG Documentation](https://microsoft.github.io/graphrag)
- [GraphRAG Research Paper](https://arxiv.org/pdf/2404.16130)

---

**Disclaimer**: This repository is a medical domain adaptation and community project. It is not affiliated with, endorsed by, or officially supported by Microsoft Corporation.
