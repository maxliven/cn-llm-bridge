# Changelog

All notable changes to cn-llm-bridge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- CI badge, Mermaid architecture diagram, and modules table in README
- Comprehensive pyproject.toml with classifiers, keywords, urls, scripts
- Ecosystem footer linking to cc-skill-router

### Changed
- Upgraded build backend from setuptools to hatchling

## [1.0.0] - 2026-07-05

### Added
- `cn_llm_bridge`: Qwen3.7-Plus vision analysis + qwen3-asr-flash cloud transcription + faster-whisper local fallback
- `kimi_bridge`: Kimi K2.7 Code deep reasoning and cross-modal synthesis
- `vision_analyze`: structured JSON image analysis with auto detail level detection
- `vision_chat`: multi-turn vision conversation
- `audio_transcribe`: audio-to-text with cloud-first + local fallback pipeline
- `kimi_chat`: deep reasoning chat with Kimi K2.7 Code
- `kimi_synthesize`: multi-source cross-modal synthesis
- `tools_health` / `kimi_health`: model availability checks
- Configurable model IDs via environment variables
- Adaptive max_tokens based on task type
- Single automatic retry on 5xx errors
- Complexity self-assessment in vision_analyze

### Fixed
- Updated Kimi model from deprecated `kimi-k2-thinking` to `kimi-k2.7-code`
