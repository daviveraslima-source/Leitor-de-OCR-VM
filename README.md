# Leitor de OCR VM Greenish

Projeto experimental para identificação de **REF** e **COR** em etiquetas Greenish.

- **REF**: código presente na etiqueta, preservando os caracteres.
- **COR**: código de exatamente quatro dígitos.
- Referências e combinações de cores são verificadas com um catálogo **privado**, fornecido no ambiente de execução.
- Leitura não confirmada deve ser marcada **REVISAR**, nunca preenchida por adivinhação.

## Segurança dos dados
O repositório é público por decisão do proprietário. **Não enviar fotos, vídeos, listas comerciais, catálogos internos nem tokens para este repositório.** Os conjuntos de dados de treinamento são carregados separadamente em armazenamento protegido no Railway.

## Estado
Repositório de código em implantação. O OCR ainda não possui precisão validada para uso sem supervisão. O site VM original no Cloudflare não é modificado.
