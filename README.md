# Actividad Evaluativa 2 — Paper corto de investigación aplicada

**Autores:** Mateo Rubio y Diego Polanco · **Asignatura:** Inteligencia Artificial · **Docente:** José Ordóñez Córdoba · Universidad ICESI

**Tema:** supervisión humana explícita, explicabilidad y control de sesgos en LLM aplicados a salud.
**Artículo base:** Bélisle-Pipon, J.-C. (2024). *Why we need to be careful with LLMs in medicine*. Frontiers in Medicine, 11, 1495582. https://doi.org/10.3389/fmed.2024.1495582

## Entregables

| Archivo | Qué es |
|---|---|
| `paper.docx` | **Entregable principal.** Portada + cuerpo de **3 páginas** + referencias APA + Apéndice de Transparencia IA. Ecuaciones nativas (editables en Word y Google Docs). |
| `paper.pdf` | El mismo documento exportado desde Word, como referencia del formato final. |
| `paper.md` | Texto fuente del paper (misma versión que el .docx). |
| `demo_colab.ipynb` | Demo para presentar en Colab: explica qué hace, por qué y cómo se relaciona con el paper; 4 actos, uno por capa de la arquitectura, con salidas ya guardadas. |
| `source.md` / `plan.md` | Artículo base y rúbrica (insumos). |

## Formato y límite de páginas

- Carta, márgenes de 2,5 cm, Times New Roman 11, interlineado sencillo.
- Página 1 = portada (sin número). **Cuerpo = páginas numeradas 1 a 3** (Resumen → Conclusión). La conclusión termina a unos 19 cm de la tercera página, así que queda margen por si Google Docs pagina un poco distinto que Word.
- Referencias y apéndice empiezan en una página nueva (no cuentan para el límite).
- El cuerpo tiene **4 ecuaciones numeradas**, cada una seguida de una explicación término a término en lenguaje natural.

### Pasarlo a Google Docs

1. En Google Drive: `Nuevo → Subir archivo → paper.docx`.
2. Clic derecho sobre el archivo → `Abrir con → Documentos de Google`.
3. Revisa que las 4 ecuaciones numeradas se vean bien y que la Conclusión siga en la página 4 del documento (3.ª del cuerpo).

### Antes de entregar

- Lean el **Apéndice de Transparencia IA** y ajústenlo para que describa con exactitud su propio proceso. La rúbrica los hace responsables de cada afirmación.

## Mapa rúbrica → entregable

| Requisito (`plan.md`) | Dónde se cumple |
|---|---|
| Máximo 3 páginas | Cuerpo de 3 páginas medidas en Word (sin portada ni referencias) |
| Introducción | Sección 1: contexto clínico con evidencia empírica (Hager 2024, Asgari 2025, Omiye 2023) y tesis del artículo base |
| Desarrollo técnico (matemática y lógica) | Sección 2.1: atención y entropía cruzada = H + KL (ec. 1), cota de Kalai y Vempala, calibración · Sección 2.2: regla de arbitraje RAG + NLI (ec. 2) y valores de Shapley · Sección 2.3: regla de rechazo de Chow extendida con el sesgo de automatización (ec. 3) e igualdad de probabilidades Δ_EO (ec. 4) |
| Discusión de frontera (*trade-offs*) | Sección 3: cuatro *trade-offs* con evidencia (Huang 2024, Bansal 2021, Goh 2024, Kleinberg 2017 / Chouldechova 2017, Reglamento UE art. 14) + Tabla 1 de métricas |
| Conclusión (impacto de negocio) | Sección 4: propuesta de valor, costo de la supervisión con la ec. (3) y cumplimiento regulatorio |
| ≥ 4 fuentes primarias indexadas, APA | 28 referencias (APA 7 en español), todas citadas en el texto; las que tienen DOI se cotejaron con Crossref |
| Apéndice de Transparencia IA | Herramientas y propósito, prompt más complejo y método de verificación |

## Cómo presentar la demo (≈ 5 min)

1. Abran [colab.research.google.com](https://colab.research.google.com) → `Archivo → Subir notebook` → `demo_colab.ipynb`.
2. `Entorno de ejecución → Ejecutar todas` (≈ 10 s; solo usa `numpy`, `pandas`, `scikit-learn` y `matplotlib`, que ya vienen en Colab).
3. Empiecen por la primera celda: explica qué hace la demo, por qué y cómo se acomoda al paper, e incluye un guion desplegable. Cada acto sigue la estructura *qué hace → por qué → dónde está en el paper → resultado → mensaje clave*. Las celdas de código muestran solo su título (doble clic para ver el código), y los actos 2, 3 y 4 tienen deslizadores.

| Acto | Mensaje | Resultado que se ve |
|---|---|---|
| 1 · Calibración | Sin calibrar, la confianza no es probabilidad | ECE 0,096 → 0,013 |
| 2 · Verificación (ec. 2) | La similitud no distingue «25 mg» de «250 mg» | Falsas que pasan: 5/5 → 1/5 con verificador |
| 2b · Shapley | La explicación describe al modelo y delata un *proxy* | `zona_residencia` es lo que más empuja el riesgo |
| 3 · Supervisión (ec. 3) | El umbral sale de los costos; ρ bajo = supervisión nominal | t* 0,77 vs. 0,75 teórico; error 18,1 % → 5,3 % |
| 4 · Equidad (ec. 4) | Cerrar una brecha abre otra | Δ_EO 0,058 → 0,030, pero Δ_PPV 0,093 → 0,108 |

> ⚠️ Los datos de la demo son **sintéticos** (semilla fija). Muestran el mecanismo y cómo se mide, no un desempeño clínico.
