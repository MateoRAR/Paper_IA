# De la «alucinación» al *bullshit*

**Supervisión humana explícita, explicabilidad y control de sesgos para LLM en salud**

Paper corto de investigación aplicada · Actividad Evaluativa 2

| | |
|---|---|
| **Autores** | Mateo Rubio y Diego Polanco |
| **Asignatura** | Inteligencia Artificial |
| **Docente** | José Ordóñez Córdoba |
| **Institución** | Universidad ICESI |
| **Fecha** | Octubre de 2026 |
| **Artículo base** | Bélisle-Pipon, J.-C. (2024). *Why we need to be careful with LLMs in medicine*. Frontiers in Medicine, 11, 1495582. https://doi.org/10.3389/fmed.2024.1495582 |

---

## Resumen

Los modelos de lenguaje de gran escala (LLM) se proponen como apoyo clínico, pero se entrenan para producir texto verosímil, no verdadero. El artículo base llama a sus errores *bullshit*: indiferencia estructural hacia la evidencia. Este trabajo convierte esa crítica en una **arquitectura de tres capas**:

1. **Generación calibrada:** la confianza del modelo debe significar lo que dice.
2. **Verificación y explicación:** cada afirmación se contrasta con evidencia curada.
3. **Supervisión humana y control de sesgos:** un profesional decide cuando hace falta y se auditan las brechas entre grupos.

El paper explica la base matemática de cada capa, discute sus límites con evidencia empírica reciente y concluye con su impacto de negocio. Una demo en Colab implementa y mide cada capa.

---

## Entregables

| Archivo | Contenido |
|---|---|
| [`paper.docx`](paper.docx) | **Paper final.** Portada, cuerpo de 3 páginas, 28 referencias en APA 7 y Apéndice de Transparencia IA. Se abre en Word o en Google Docs. |
| [`paper.pdf`](paper.pdf) | El mismo paper exportado a PDF. |
| [`demo_colab.ipynb`](demo_colab.ipynb) | Demo reproducible en Google Colab, con los resultados ya guardados. |
| [`presentacion.pptx`](presentacion.pptx) | Presentación de 28 diapositivas con notas del orador. |
| [`paper.md`](paper.md) | Texto fuente del paper en Markdown. |
| [`plan.md`](plan.md) · [`source.md`](source.md) | Rúbrica de la actividad y texto del artículo base. |

---

## 1. Paper

**Formato:** carta, márgenes de 2,5 cm, Times New Roman 11, interlineado sencillo. La portada no lleva número. El cuerpo ocupa las páginas 1 a 3. Las referencias y el apéndice empiezan en una página nueva.

| Sección | Contenido |
|---|---|
| Resumen | Tesis y aporte del trabajo |
| 1. Introducción | Evidencia de fallos de los LLM en clínica y reencuadre de «alucinación» a *bullshit* |
| 2. Desarrollo técnico | Las tres capas de la arquitectura y su base matemática |
| 3. Discusión de frontera | Cuatro *trade-offs* y la Tabla 1 de métricas operativas |
| 4. Conclusión | Impacto en contextos reales de negocio |
| Referencias | 28 fuentes primarias en APA 7 |
| Apéndice de Transparencia IA | Herramientas y propósito, prompt más complejo y método de verificación |

**Ecuaciones.** El cuerpo tiene cuatro ecuaciones numeradas. Cada una va seguida de una explicación en lenguaje natural.

| Ec. | Qué expresa |
|---|---|
| (1) | La entropía cruzada se divide en entropía más divergencia KL: entrenar enseña a imitar texto, no a decir la verdad |
| (2) | Una afirmación se entrega solo si trata el mismo tema que la evidencia **y** dice lo mismo (similitud e implicación lógica, NLI) |
| (3) | Se escala a un profesional cuando la confianza calibrada es menor que 1 − c_h/(ρ·c_e) |
| (4) | La brecha de igualdad de probabilidades (Δ_EO) compara errores entre grupos |

**Fuentes.** Las 28 referencias están citadas en el texto. Las que tienen DOI se cotejaron con los metadatos de Crossref, y las cifras citadas se contrastaron con el resumen original de cada estudio.

---

## 2. Demo en Colab

**Cómo ejecutarla**

1. Abrir [colab.research.google.com](https://colab.research.google.com) → `Archivo → Subir notebook` → `demo_colab.ipynb`.
2. `Entorno de ejecución → Ejecutar todas`.

Solo usa `numpy`, `pandas`, `scikit-learn` y `matplotlib`, que ya vienen instalados en Colab. Los datos son **sintéticos** y se generan con semilla fija, así que los resultados son siempre los mismos. La demo muestra el mecanismo de cada capa y cómo se mide, no el desempeño clínico de un modelo real.

**Estructura.** La primera celda explica qué hace la demo, por qué y cómo se relaciona con el paper. Cada acto incluye su descripción, su objetivo, el resultado y un mensaje clave.

| Parte | Tema | Sección del paper | Resultado |
|---|---|---|---|
| 0 | Datos sintéticos y modelo de riesgo | Base de los actos | 20 000 pacientes ficticios |
| 1 | Calibración de la confianza | 2.1 | ECE de 0,096 a 0,013 |
| 2 | Verificación contra evidencia | 2.2 · ec. (2) | Afirmaciones falsas que pasan: de 5 a 1 |
| 2b | Explicación con valores de Shapley | 2.2 | La variable que más pesa es un *proxy* no clínico |
| 3 | Supervisión humana | 2.3 · ec. (3) | Error que llega al paciente: de 18,1 % a 5,3 % |
| 4 | Auditoría de sesgos | 2.3 · ec. (4) | Δ_EO baja de 0,058 a 0,030, pero sube la brecha de acierto de las alarmas |
| 5 | Panel de control | Tabla 1 | Todas las métricas en una tabla |

---

## 3. Presentación

28 diapositivas, cada una con notas del orador.

| Diapositivas | Bloque | Contenido |
|---|---|---|
| 1 a 3 | Introducción | Portada, el problema en cifras y el reencuadre de «alucinación» a *bullshit* |
| 4 a 10 | Arquitectura | Las tres capas, sus ecuaciones y los resultados de la demo, con una guía para leer cada gráfico |
| 11 y 12 | Discusión | Cuatro *trade-offs* y la organización de la demo |
| 13 | Conclusión | Impacto de negocio |
| 14 a 27 | Repaso | 7 preguntas de opción múltiple, cada una seguida de su respuesta y una explicación |
| 28 | Cierre | Referencias clave y espacio para preguntas |

---

## Cumplimiento de la rúbrica

| Requisito | Cómo se cumple |
|---|---|
| Paper corto de máximo 3 páginas | El cuerpo ocupa 3 páginas, sin contar portada ni referencias |
| Introducción | Contexto clínico con evidencia empírica y tesis del artículo base |
| Desarrollo técnico | Componentes matemáticos y lógicos de las tres capas, en cuatro ecuaciones explicadas |
| Discusión de frontera | Cuatro *trade-offs* sustentados en estudios primarios y una tabla de métricas y señales de alarma |
| Conclusión | Propuesta de valor, costo de la supervisión y cumplimiento regulatorio |
| Mínimo 4 fuentes primarias en APA | 28 referencias en APA 7, todas citadas en el texto |
| Apéndice de Transparencia IA | Herramientas y propósito, ejemplo del prompt más complejo y método de auditoría |
| **Rigor técnico (30 %)** | Ecuaciones derivadas y verificadas, y afirmaciones matizadas con la literatura |
| **Uso crítico de IAG (25 %)** | Uso de la IA en cinco tareas declaradas y verificación independiente de fuentes y cifras |
| **Estructura y argumentación (25 %)** | Una tesis que se desarrolla capa por capa y se valida con la demo |
| **Calidad de fuentes (20 %)** | Literatura primaria revisada por pares (Nature, Science, Radiology, JAMA Network Open, NeurIPS, ACL, STOC, entre otras) |
