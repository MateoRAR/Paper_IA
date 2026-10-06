# De la «alucinación» al *bullshit*: supervisión humana explícita, explicabilidad y control de sesgos para LLM en salud

Mateo Rubio y Diego Polanco
Universidad ICESI
Inteligencia Artificial — Actividad Evaluativa 2: paper corto de investigación aplicada
Docente: José Ordóñez Córdoba
Octubre de 2026

<!-- fin-portada -->

**Resumen.** Los modelos de lenguaje de gran escala (LLM) se proponen como apoyo clínico, pero se entrenan para producir texto verosímil, no verdadero. Siguiendo a Hicks et al. (2024), Bélisle-Pipon (2024) llama a sus errores *bullshit*: indiferencia estructural hacia la evidencia. Este trabajo convierte esa crítica en una arquitectura de tres capas —generación calibrada, verificación con evidencia y supervisión humana con auditoría de sesgos—, explica su base matemática, discute sus límites y la ilustra con un *notebook* reproducible.

**Palabras clave:** LLM, alucinación, supervisión humana, explicabilidad, equidad algorítmica, salud.

## 1. Introducción

Los LLM aprueban preguntas tipo examen de licencia médica (Singhal et al., 2023), pero fallan en condiciones realistas: en 2.400 casos reales diagnosticaron peor que los médicos y no siguieron las guías clínicas (Hager et al., 2024). Aun al resumir consultas se han medido un 1,47 % de oraciones alucinadas y un 3,45 % de omisiones (Asgari et al., 2025), y los modelos comerciales repiten mitos de la medicina basada en la raza (Omiye et al., 2023).

El artículo base (Bélisle-Pipon, 2024) sostiene que llamar «alucinaciones» a estos errores es engañoso, porque sugiere una percepción fallida y, por tanto, corregible. Siguiendo a Hicks et al. (2024) y a Frankfurt (2005), propone llamarlos *bullshit*: texto plausible producido con indiferencia hacia su verdad. Si el error es estructural, no basta con un mejor modelo: hay que contenerlo con un sistema. El artículo, sin embargo, deja sus propuestas —capas de verificación, explicabilidad (XAI) y supervisión— en el plano normativo.

Este trabajo las especifica: muestra por qué el entrenamiento no apunta a la verdad, define criterios operativos para verificar y explicar, modela la supervisión humana como una decisión de costos con auditoría de equidad, y contrasta cada capa con la evidencia empírica sobre sus límites.

## 2. Desarrollo técnico: la arquitectura

La arquitectura reemplaza el uso del LLM como oráculo por un flujo de tres capas en el que cada salida termina en una de dos acciones: se entrega acompañada de su evidencia o se escala a un profesional.

### 2.1 Capa generativa: el objetivo no es la verdad

Un Transformer (Vaswani et al., 2017) procesa el texto mediante atención: cada fragmento de palabra (*token*) se representa como un promedio ponderado de los demás, según cuán relacionados están. Con esa representación estima la probabilidad del siguiente *token* dado lo anterior, $p_\theta(x_t\mid x_{<t})$, donde $\theta$ son sus parámetros, y se preentrena minimizando la entropía cruzada:

$$\mathcal{L}(\theta)=\mathbb{E}_{x\sim p_{\mathrm{data}}}\left[-\log\,p_\theta(x)\right]=H(p_{\mathrm{data}})+\mathrm{KL}(p_{\mathrm{data}}\,\|\,p_\theta)$$ (1)

La pérdida $\mathcal{L}$ mide cuánto «se sorprende» el modelo ante el texto real y tiene dos términos: la entropía $H(p_{\mathrm{data}})$, que mide cuán impredecible es el lenguaje y no depende del modelo, y la divergencia de Kullback-Leibler (KL), que mide cuánto se aleja la distribución del modelo, $p_\theta$, de la del texto, $p_{\mathrm{data}}$. Entrenar solo reduce el segundo término: enseña a imitar el texto, no a contrastarlo con los hechos, de modo que una afirmación falsa pero frecuente es «correcta» para el objetivo. El ajuste con retroalimentación humana (RLHF) optimiza la preferencia de evaluadores, que tampoco es la verdad (Ouyang et al., 2022). Kalai y Vempala (2024) lo formalizan: un modelo calibrado debe alucinar hechos arbitrarios a una tasa cercana a la fracción de hechos que aparecen una sola vez en el corpus, aun con datos perfectos. La cota, con todo, no aplica a hechos frecuentes o sistemáticos, lo que matiza la tesis del artículo base.

Por eso, la confianza del modelo debe calibrarse antes de usarse para decidir: si declara 80 %, debe acertar el 80 % de las veces. Se mide con el error de calibración esperado (ECE), la brecha promedio entre confianza y acierto, y se corrige con *temperature scaling*, que divide las puntuaciones del modelo por una constante ajustada en validación (Guo et al., 2017). En respuestas abiertas es mejor la entropía semántica: si varias respuestas muestreadas difieren en significado, el modelo probablemente está confabulando (Farquhar et al., 2024).

### 2.2 Capa de verificación y explicabilidad

La segunda capa descompone la salida en afirmaciones $a$ y las contrasta con una base de evidencia curada $\mathcal{D}$ (guías clínicas, no la web), como en la generación aumentada por recuperación (RAG; Lewis et al., 2020), que en medicina mejora la factualidad (Zakka et al., 2024). Para cada afirmación se recupera el documento más parecido, $d^{*}$, y se aplica la regla:

$$\mathrm{entregar}(a)\iff\cos(e(a),e(d^{*}))\geq\tau_s\;\wedge\;P_{\mathrm{NLI}}(d^{*}\Rightarrow a)\geq\tau_e$$ (2)

La regla exige dos condiciones. Primero, que $a$ y $d^{*}$ traten el mismo tema: la similitud coseno entre sus representaciones vectoriales $e(\cdot)$ debe superar el umbral $\tau_s$. Segundo, que digan lo mismo: un modelo de inferencia en lenguaje natural (NLI) debe estimar, con probabilidad de al menos $\tau_e$, que la evidencia implica la afirmación. Si alguna falla, o si la incertidumbre de la capa 1 es alta, la afirmación se escala. La segunda condición es indispensable: «25 mg/kg» y «250 mg/kg» son casi idénticas para la similitud, pero contradictorias para un NLI, como muestra el *notebook*.

La explicabilidad tiene dos niveles. Los pesos de atención son una pista, no una explicación: atenciones muy distintas pueden producir la misma predicción (Jain y Wallace, 2019). La atribución post-hoc usa valores de Shapley (Shapley, 1953; Lundberg y Lee, 2017): la contribución de cada variable es su efecto marginal promedio sobre todos los órdenes en que las variables podrían incorporarse, y las contribuciones suman exactamente la diferencia entre la predicción del caso y la predicción promedio (propiedad de eficiencia). Ambos niveles describen qué patrones movieron la salida, no por qué sería clínicamente correcta (Ghassemi et al., 2021); por eso, la explicación que ve el clínico es la evidencia $d^{*}$.

### 2.3 Capa de supervisión humana y control de sesgos

La supervisión se formaliza como una opción de rechazo (Chow, 1970): ante cada salida, el sistema elige entre entregarla o escalarla a un profesional según cuál tenga menor costo esperado. Sea $\pi$ la probabilidad calibrada de que la salida sea correcta, $c_e$ el costo de un error que llega al paciente, $c_h$ el de una revisión y $\rho$ la fracción de errores que el revisor detecta (sin introducir errores nuevos). Entregar cuesta, en promedio, $(1-\pi)c_e$; escalar cuesta la revisión más los errores que se le escapan al revisor, $c_h+(1-\pi)(1-\rho)c_e$. Escalar conviene cuando el segundo costo es menor que el primero; al despejar $\pi$:

$$\pi<1-\frac{c_h}{\rho\,c_e}$$ (3)

Así, se escala toda salida cuya confianza quede por debajo de ese umbral. Si revisar cuesta la cuarta parte de un error ($c_h/c_e=0{,}25$) y el revisor detecta todos los errores ($\rho=1$, que es la regla de Chow), se escala lo que tenga menos de 75 % de confianza; si solo detecta la mitad ($\rho=0{,}5$), el umbral cae a 50 %, revisar deja de compensar y la supervisión se vuelve nominal. Este sesgo de automatización está documentado: la exactitud de radiólogos inexpertos fue de 79,7 % con sugerencias correctas de un supuesto sistema de IA y de 19,8 % con sugerencias erróneas; la de expertos, de 82,3 % y 45,5 % (Dratsch et al., 2023). Por eso $\rho$ debe medirse con casos de control.

El control de sesgos audita métricas por grupo. Con un atributo sensible $A$, el resultado real $Y$ y la predicción $\hat{Y}$, la paridad demográfica compara la proporción de pacientes marcados en cada grupo, y la igualdad de probabilidades (*equalized odds*) compara sus errores:

$$\Delta_{\mathrm{EO}}=\max\limits_{y\in\{0,1\}}\left|P(\hat{Y}=1\mid Y=y,A=0)-P(\hat{Y}=1\mid Y=y,A=1)\right|$$ (4)

Con $y=1$ se comparan las tasas de verdaderos positivos de ambos grupos (¿detecta por igual a quienes tienen el evento?); con $y=0$, las de falsos positivos (¿alarma por igual a quienes no lo tienen?). $\Delta_{\mathrm{EO}}$ es la peor de las dos brechas y vale cero si el sistema se equivoca igual en ambos grupos. El posprocesamiento de Hardt et al. (2016) elige un umbral de decisión por grupo que minimiza la pérdida con $\Delta_{\mathrm{EO}}\leq\varepsilon$. Omitir $A$ no basta: un algoritmo que usaba el gasto sanitario como *proxy* de la necesidad clínica subestimó la gravedad de los pacientes negros (Obermeyer et al., 2019).

## 3. Discusión de frontera: *trade-offs*

**Más modelos no producen más verdad.** Verificar un LLM con otro —«combatir fuego con fuego», según el artículo base— comparte el objetivo (1) y puede propagar el error: sin retroalimentación externa, los LLM no logran autocorregir su razonamiento (Huang et al., 2024). Lo que cambia el problema es la fuente externa $\mathcal{D}$, a costa de cobertura: lo que no está en $\mathcal{D}$ se escala, y una base desactualizada se vuelve otra fuente de error.

**Explicar no es justificar.** Las explicaciones aumentan la aceptación de las recomendaciones de la IA tanto si son correctas como si no (Bansal et al., 2021). La XAI sirve para rendir cuentas, pero es insuficiente como salvaguarda.

**El humano en el circuito no es automáticamente complementario.** En un ensayo aleatorizado con 50 médicos, el acceso a GPT-4 no mejoró significativamente su razonamiento diagnóstico (76 % frente a 74 %), aunque el modelo por sí solo superó en 16 puntos al grupo control (Goh et al., 2024). El valor de la supervisión depende de su diseño —qué se escala, con qué evidencia y cómo se mide $\rho$—. Por eso, el Reglamento europeo de IA exige que quien supervisa un sistema de alto riesgo pueda advertir el sesgo de automatización y anular su salida (Reglamento [UE] 2024/1689, 2024, art. 14).

**La equidad no es un escalar.** Si las tasas base difieren entre grupos, ningún clasificador imperfecto puede estar calibrado por grupo e igualar a la vez las tasas de falsos positivos y de falsos negativos (Kleinberg et al., 2017; Chouldechova, 2017). Elegir entre paridad demográfica, igualdad de probabilidades o calibración por grupo es una decisión de valor que debe declararse. La Tabla 1 resume las métricas con las que se monitoriza la arquitectura.

**Tabla 1.** Métricas operativas y señales de alarma por capa.

| Capa | Métrica | Señal de alarma |
|---|---|---|
| Generativa | ECE; entropía semántica | Alta confianza con bajo acierto |
| Verificación | % de afirmaciones sin soporte en $\mathcal{D}$; precisión del NLI | Contradicciones que superan el filtro |
| Explicabilidad | Fidelidad por perturbación | Explicación estable pero no fiel |
| Supervisión | Tasa de escalado; $\rho$ (ec. 3); tasa de anulación | El revisor aprueba más del 95 % |
| Equidad | Paridad demográfica; $\Delta_{\mathrm{EO}}$ (ec. 4); calibración por grupo | Brechas persistentes entre grupos |

## 4. Conclusión

El aporte del artículo base es un cambio de marco; el de este trabajo, convertirlo en requisitos verificables: confianza calibrada, afirmaciones ancladas a evidencia curada y verificadas por implicación lógica, escalado humano con un umbral explícito y auditoría de equidad declarada. La mitigación no vive en el modelo, sino en la arquitectura que lo envuelve.

Para el negocio, esto redefine la propuesta de valor: no se vende «diagnóstico automático», sino menos carga documental con riesgo acotado y trazable. Con un marco de evaluación clínica, los errores graves de las notas generadas pueden quedar por debajo de los reportados para la documentación manual (Asgari et al., 2025). La ecuación (3) traduce el despliegue a términos de gestión: conocidos $c_h/c_e$ y $\rho$, la fracción escalada y el error residual son cuantificables, y la revisión humana deja de ser un sobrecosto para convertirse en el precio de la responsabilidad legal y del cumplimiento regulatorio. Un sistema que no mide $\rho$ ni sus brechas de equidad no está listo para producción, por convincente que suene.

<!-- pagebreak -->

## Referencias

Asgari, E., Montaña-Brown, N., Dubois, M., Khalil, S., Balloch, J., Yeung, J. A. y Pimenta, D. (2025). A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation. *npj Digital Medicine, 8*, Artículo 274. https://doi.org/10.1038/s41746-025-01670-7

Bansal, G., Wu, T., Zhou, J., Fok, R., Nushi, B., Kamar, E., Ribeiro, M. T. y Weld, D. (2021). Does the whole exceed its parts? The effect of AI explanations on complementary team performance. En *Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems* (pp. 1–16). ACM. https://doi.org/10.1145/3411764.3445717

Bélisle-Pipon, J.-C. (2024). Why we need to be careful with LLMs in medicine. *Frontiers in Medicine, 11*, Artículo 1495582. https://doi.org/10.3389/fmed.2024.1495582

Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big Data, 5*(2), 153–163. https://doi.org/10.1089/big.2016.0047

Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory, 16*(1), 41–46. https://doi.org/10.1109/TIT.1970.1054406

Dratsch, T., Chen, X., Rezazade Mehrizi, M., Kloeckner, R., Mähringer-Kunz, A., Püsken, M., Baeßler, B., Sauer, S., Maintz, D. y Pinto dos Santos, D. (2023). Automation bias in mammography: The impact of artificial intelligence BI-RADS suggestions on reader performance. *Radiology, 307*(4), Artículo e222176. https://doi.org/10.1148/radiol.222176

Farquhar, S., Kossen, J., Kuhn, L. y Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature, 630*(8017), 625–630. https://doi.org/10.1038/s41586-024-07421-0

Frankfurt, H. G. (2005). *On bullshit*. Princeton University Press.

Ghassemi, M., Oakden-Rayner, L. y Beam, A. L. (2021). The false hope of current approaches to explainable artificial intelligence in health care. *The Lancet Digital Health, 3*(11), e745–e750. https://doi.org/10.1016/S2589-7500(21)00208-9

Goh, E., Gallo, R., Hom, J., Strong, E., Weng, Y., Kerman, H., Cool, J. A., Kanjee, Z., Parsons, A. S., Ahuja, N., Horvitz, E., Yang, D., Milstein, A., Olson, A. P. J., Rodman, A. y Chen, J. H. (2024). Large language model influence on diagnostic reasoning: A randomized clinical trial. *JAMA Network Open, 7*(10), Artículo e2440969. https://doi.org/10.1001/jamanetworkopen.2024.40969

Guo, C., Pleiss, G., Sun, Y. y Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning, PMLR 70*, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html

Hager, P., Jungmann, F., Holland, R., Bhagat, K., Hubrecht, I., Knauer, M., Vielhauer, J., Makowski, M., Braren, R., Kaissis, G. y Rueckert, D. (2024). Evaluation and mitigation of the limitations of large language models in clinical decision-making. *Nature Medicine, 30*(9), 2613–2622. https://doi.org/10.1038/s41591-024-03097-1

Hardt, M., Price, E. y Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems, 29*, 3315–3323.

Hicks, M. T., Humphries, J. y Slater, J. (2024). ChatGPT is bullshit. *Ethics and Information Technology, 26*(2), Artículo 38. https://doi.org/10.1007/s10676-024-09775-5

Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X. y Zhou, D. (2024). Large language models cannot self-correct reasoning yet. En *The Twelfth International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=IkmD3fKBPQ

Jain, S. y Wallace, B. C. (2019). Attention is not explanation. En *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (Vol. 1, pp. 3543–3556). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1357

Kalai, A. T. y Vempala, S. S. (2024). Calibrated language models must hallucinate. En *Proceedings of the 56th Annual ACM Symposium on Theory of Computing* (pp. 160–171). ACM. https://doi.org/10.1145/3618260.3649777

Kleinberg, J., Mullainathan, S. y Raghavan, M. (2017). Inherent trade-offs in the fair determination of risk scores. En C. H. Papadimitriou (Ed.), *8th Innovations in Theoretical Computer Science Conference (ITCS 2017)* (LIPIcs, Vol. 67, pp. 43:1–43:23). Schloss Dagstuhl – Leibniz-Zentrum für Informatik. https://doi.org/10.4230/LIPIcs.ITCS.2017.43

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. y Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474.

Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*, 4765–4774.

Obermeyer, Z., Powers, B., Vogeli, C. y Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science, 366*(6464), 447–453. https://doi.org/10.1126/science.aax2342

Omiye, J. A., Lester, J. C., Spichak, S., Rotemberg, V. y Daneshjou, R. (2023). Large language models propagate race-based medicine. *npj Digital Medicine, 6*, Artículo 195. https://doi.org/10.1038/s41746-023-00939-z

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J. y Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems, 35*, 27730–27744.

Reglamento (UE) 2024/1689 del Parlamento Europeo y del Consejo, de 13 de junio de 2024, por el que se establecen normas armonizadas en materia de inteligencia artificial (Reglamento de Inteligencia Artificial). (2024). *Diario Oficial de la Unión Europea*, L 2024/1689. http://data.europa.eu/eli/reg/2024/1689/oj

Shapley, L. S. (1953). A value for n-person games. En H. W. Kuhn y A. W. Tucker (Eds.), *Contributions to the theory of games* (Vol. 2, pp. 307–318). Princeton University Press. https://doi.org/10.1515/9781400881970-018

Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., Payne, P., Seneviratne, M., Gamble, P., Kelly, C., Babiker, A., Schärli, N., Chowdhery, A., Mansfield, P., Demner-Fushman, D., … Natarajan, V. (2023). Large language models encode clinical knowledge. *Nature, 620*(7972), 172–180. https://doi.org/10.1038/s41586-023-06291-2

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. y Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*, 5998–6008.

Zakka, C., Shad, R., Chaurasia, A., Dalal, A. R., Kim, J. L., Moor, M., Fong, R., Phillips, C., Alexander, K., Ashley, E., Boyd, J., Boyd, K., Hirsch, K., Langlotz, C., Lee, R., Melia, J., Nelson, J., Sallam, K., Tullis, S., … Hiesinger, W. (2024). Almanac — Retrieval-augmented language models for clinical medicine. *NEJM AI, 1*(2). https://doi.org/10.1056/AIoa2300068

## Apéndice de Transparencia IA

**Herramientas y propósito.** Se usaron dos asistentes de IA generativa: *DeepSeek V4.1 Flash* (vía OpenCode), en la primera fase, y *Claude Code* (modelo Claude Opus 5.5, de Anthropic), en la segunda. Su uso se concentró en cinco tareas: (1) *Buscar fuentes:* localizar literatura primaria indexada sobre alucinación, calibración, supervisión humana y equidad, y verificar sus metadatos (DOI, autores, volumen y páginas). (2) *Resolver dudas:* aclarar conceptos técnicos antes de escribirlos, como la descomposición de la entropía cruzada en entropía más divergencia KL, la regla de rechazo de Chow o el teorema de imposibilidad de Kleinberg et al. (3) *Corregir errores:* revisar el borrador para detectar citas incorrectas, afirmaciones exageradas y fórmulas mal enunciadas, y depurar el código del *notebook*. (4) *Evaluar trade-offs:* contrastar alternativas de diseño de la arquitectura (similitud frente a verificación por NLI, umbral único frente a umbrales por grupo) y del propio texto (qué ecuaciones conservar dentro del límite de tres páginas). (5) *Concretar objetivos:* delimitar las contribuciones del paper y el alcance de la demo. También se usaron como apoyo para dar formato al documento. No se usó IA para inventar datos: las cifras empíricas citadas provienen de las fuentes referenciadas, y las del *notebook* son sintéticas y se declaran como tales. Los autores son los únicos responsables de la veracidad de cada afirmación.

**Ejemplo del prompt más complejo diseñado.** Corresponde a la tarea (4), evaluar *trade-offs*: «Actúa como revisor técnico de un paper sobre LLM en salud. Te comparto la arquitectura de tres capas de nuestro borrador: calibración de la confianza, verificación con RAG e implicación lógica (NLI), y supervisión humana con auditoría de sesgos. Para cada capa, identifica su principal *trade-off*: qué gana el sistema, qué pierde y en qué condiciones la capa falla. Sustenta cada *trade-off* con al menos una fuente primaria revisada por pares e indica su DOI; si no encuentras una fuente verificable, dilo explícitamente en lugar de inventarla. Distingue lo que la fuente demuestra empíricamente de lo que es una inferencia nuestra. Luego propón, para cada capa, una métrica que permita monitorear ese *trade-off* en producción y el valor a partir del cual sería una señal de alarma. Restricciones: máximo 200 palabras por capa, ninguna cifra que no esté en las fuentes y señala cualquier punto en que tus conclusiones contradigan la tesis del artículo base (Bélisle-Pipon, 2024).»

**Método de auditoría y verificación.** (a) *Fuentes:* cada referencia con DOI se contrastó con los metadatos de Crossref (autores, título, volumen y páginas), y cada cifra citada (p. ej., 79,7 % frente a 19,8 % en Dratsch et al.; 1,47 % en Asgari et al.; 76 % frente a 74 % en Goh et al.) se cotejó con el resumen original en PubMed/Europe PMC o en la revista. Esta revisión detectó y corrigió un error de la primera versión: el título y el año del artículo base estaban mal citados. (b) *Matemática:* cada ecuación se derivó a mano; en particular, (3) se obtiene al comparar $c_h+(1-\pi)(1-\rho)c_e$ con $(1-\pi)c_e$, y se comprobó que con $\rho=1$ coincide con la regla de Chow (1970). (c) *Empírica:* el *notebook* se ejecutó de extremo a extremo; comprueba numéricamente la eficiencia de Shapley (error del orden de $10^{-17}$) y que el umbral óptimo hallado por búsqueda coincide con (3) salvo ruido de muestreo (0,77 frente a 0,75, con costos esperados prácticamente iguales). (d) *Conceptual:* se corrigieron afirmaciones excesivas del borrador (p. ej., que la atención «permite rastrear» qué sostiene la salida), contrastándolas con Jain y Wallace (2019), y se matizó la tesis del artículo base con Kalai y Vempala (2024).
