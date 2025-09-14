# Trabalho de Graduação — Engenharia de Informação (UFABC)

Este repositório contém o projeto desenvolvido como parte do **Trabalho de Graduação** em Engenharia de Informação na **Universidade Federal do ABC (UFABC)**.  
O projeto explora soluções em **visão computacional** e **aprendizado de máquina** aplicadas ao reconhecimento de informações em dispositivos de medição médica, como oxímetros e medidores de pressão.

---

## 📂 Estrutura do Repositório

| Pasta / Arquivo | Descrição |
|-----------------|-----------|
| `modelo_sete_segmento/` | Implementação e testes do modelo baseado em display de sete segmentos. |
| `poc_ml_kit/` | Prova de conceito utilizando o **ML Kit** do Google para reconhecimento de texto/imagem. |
---

## 🎯 Objetivos do Projeto

- Desenvolver um sistema para **captura e reconhecimento de dados clínicos** a partir de dispositivos com display digital.  
- Implementar e comparar diferentes abordagens:  
  - **Rede neural convolucional (CNN)** para leitura de dígitos em displays de sete segmentos.  
  - **Google ML Kit** como ferramenta de reconhecimento embarcado em dispositivos móveis.  
- Validar a viabilidade de uso dessas soluções em um contexto de **telemedicina** e acompanhamento remoto de pacientes.  

---

## ⚙️ Tecnologias Utilizadas

- **Linguagens:** Kotlin, Python
- **Frameworks/Bibliotecas:**  
  - [TensorFlow / TensorFlow Lite](https://www.tensorflow.org/)  
  - [OpenCV](https://opencv.org/)  
  - [Google ML Kit](https://developers.google.com/ml-kit)  
- **Ambiente de desenvolvimento:** Android Studio, ferramentas de build Gradle, compiladores g++/clang.  

---

## 🚀 Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/favelar86/trabalho-graduacao-engenharia-info.git
cd trabalho-graduacao-engenharia-info
