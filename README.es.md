# ¿Podar, cuantizar o ambos? Curvas precisión–tamaño honestas para un Vision Transformer

*[English version](README.md)*

## Resumen

La compresión de Vision Transformers suele reportarse con cifras teóricas, como bits por peso o número de parámetros, y no con el tamaño real del modelo. Cada método publicado usa su propio protocolo (distinto número de imágenes de calibración y distinta precisión de referencia del modelo sin comprimir), por lo que sus resultados no son directamente comparables entre sí. Rara vez se comparan con un modelo más pequeño sin comprimir, y casi nunca se evalúan fuera de ImageNet-val, por ejemplo con fotos nuevas de las mismas clases (ImageNet-V2) o con otros estilos visuales (ImageNet-R).

Sobre DeiT-Small en ImageNet-1k, este trabajo pregunta en qué tramos de la curva precisión–tamaño conviene podar, cuantizar o combinar ambas técnicas, y qué componentes del modelo explican la diferencia.

Para responderlo con una medición común, cada configuración se ubica en los mismos ejes según su tamaño real en bytes, con intervalos de confianza por bootstrap pareado. Se compara con DeiT-Tiny y con métodos publicados de compresión post-entrenamiento, todos ejecutados con el código de sus autores y un protocolo común. Además, se evalúa si las conclusiones se mantienen en ImageNet-V2 e ImageNet-R.

Sobre esa base se propone un procedimiento post-entrenamiento. Mide la sensibilidad de la atención y del MLP de cada bloque a la poda estructurada y a la cuantización de pesos. Elimina cabezas de atención y neuronas del MLP compensando la capa siguiente por mínimos cuadrados, cuantiza los pesos restantes y asigna estas decisiones por bloque para alcanzar una serie de tamaños objetivo. Restringirlo a solo cuantizar o solo podar produce curvas directamente comparables entre sí.

## Qué hay aquí

Las secciones de abajo cubren los dos modelos de referencia, las líneas base densas guardadas como archivos reales y el primer método publicado ubicado en el gráfico. El trabajo posterior, los demás métodos publicados, la poda, nuestra asignación y las pruebas de activaciones, está en los cuadernos 03 a 06 y en la hoja de ruta, y su redacción en esta página viene después. Las decisiones detrás del protocolo, con sus razones, están en [`DECISIONS.es.md`](DECISIONS.es.md).

| Ruta | Rol |
|---|---|
| `01_rebuild_deit.ipynb` | DeiT reconstruido desde cero y verificado contra timm |
| `02_dense_baselines.ipynb` | DeiT-S y DeiT-Ti en FP32, FP16, INT8 e INT4 como archivos reales |
| `03_known_methods.ipynb` | los métodos de cuantización publicados, tal como se publicaron y solo pesos, frente a sus artículos |
| `04_known_pruning.ipynb` | los métodos de poda publicados, sin reentrenar |
| `05_fragility.ipynb` | qué parte de FIMA-Q es frágil, frente a las predicciones de `ABLATIONS.md` |
| `06_prune_quantize_curves.ipynb` | modelos podados a pocos bits, las tres curvas de nuestra asignación y los costos además del tamaño |
| `deitc/` | el código que comparten cuadernos y scripts: modelo, datos, archivos empaquetados, evaluación y archivos de resultados, la lectura de checkpoints publicados, costos, poda y redondeo |
| `scripts/setup_third_party.sh` | el código de los métodos publicados, clonado en los commits que usamos, con nuestros parches |
| `scripts/data/prepare_imagenet.py`, `scripts/data/make_calib_list.py` | los datos, y las imágenes de calibración que comparten todos los métodos (`splits/calib_seed0.txt`) |
| `scripts/quantize/pack_published.py`, `scripts/quantize/evaluate_file.py` | del checkpoint de un método publicado a un archivo verificado, y de un archivo a sus resultados |
| `methods/adalog/` | nuestro parche al código de AdaLog y el script que lo ejecuta |
| `methods/aphq/`, `methods/fimaq/`, `methods/repq/`, `methods/ptq4vit/` | lo mismo para APHQ-ViT, FIMA-Q, RepQ-ViT y PTQ4ViT, y las pruebas de activaciones de FIMA-Q |
| `deitc/pruning.py`, `scripts/prune/prune_points.py`, `scripts/prune/combine_points.py` | la poda sin reentrenar (nuestra reimplementación de CORP, GRAIL y la etapa de poda de VBP), y los modelos podados llevados a pocos bits |
| `scripts/allocate/sensitivity.py`, `scripts/allocate/allocate.py` | la sensibilidad de cada capa y bloque, y nuestra asignación por bloque bajo un tamaño objetivo |
| `scripts/measure/measure_costs.py`, `scripts/measure/measure_latency.py` | MACs, BOPs, latencia y memoria |
| `scripts/quantize/w8a8_torchao.py`, `scripts/quantize/a8_score.py`, `scripts/measure/gemlite_test.py`, `scripts/prune/quantize_then_prune.py` | los archivos INT8, las activaciones de 8 bits sobre modelos empaquetados, los kernels de GemLite y la comprobación del orden de compresión |
| `ABLATIONS.md` | las ablaciones y las pruebas de activaciones, con lo que esperábamos antes de ejecutarlas |
| `outputs/*_summary.json` | las cifras de esta página |

## 1. Reconstrucción de DeiT-S

DeiT-S se reimplementa en PyTorch puro, con la atención escrita explícitamente para poder quitar cabezas completas, y se carga con los pesos oficiales. La corrida del cuaderno usó una RTX 4070, torch 2.11 y timm 1.0.30.

| Verificación | Resultado |
|---|---|
| Parámetros | 22 050 664; el `state_dict` oficial carga con `strict=True` |
| Misma función que timm | predicciones idénticas en las 10 000 imágenes de ImageNet-V2 |
| Top-1 en ImageNet-V2, DeiT-S | 68.52 % [67.63; 69.48], publicado 68.53 % |
| Top-1 en ImageNet-V2, DeiT-Ti | 59.81 % [58.86; 60.84], publicado 59.92 % |
| Dónde están los pesos | 96.5 % en las capas lineales de los bloques (MLP 64.3 %, atención 32.2 %) |
| Exceso de tamaño sobre parámetros × bits / 8 | +4 % en INT8, +23 % en INT4, +32 % en INT3, por escalas, puntos cero y capas en FP16 |

Los corchetes son intervalos bootstrap del 95 %. Los tamaños INT del cuaderno 01 son calculados; desde el cuaderno 02, cada tamaño es el de un archivo en disco.

## 2. Las líneas base densas como archivos reales

Cada modelo se guarda en cada formato, se relee, se verifica bit a bit contra lo escrito y se evalúa desde el archivo. El cuantizador es redondeo simple de los pesos de las 48 capas lineales de los bloques, en grupos de 64 para INT4, con el resto en FP16. No usa datos, así que es el piso que todo método publicado debería superar.

| Modelo | Archivo (MB) | ImageNet-1k val | ImageNet-V2 | ImageNet-R |
|---|---|---|---|---|
| DeiT-S FP32 | 88.2 | 79.86 | 68.52 | 42.24 |
| DeiT-S FP16 | 44.1 | 79.86 | 68.53 | 42.25 |
| DeiT-S INT8 | 23.0 | 79.84 | 68.41 | 42.27 |
| DeiT-S INT4 | 13.6 | 79.18 | 67.76 | 41.68 |
| DeiT-Ti FP32 | 22.9 | 72.17 | 59.81 | 32.70 |
| DeiT-Ti FP16 | 11.4 | 72.17 | 59.80 | 32.70 |
| DeiT-Ti INT8 | 6.2 | 72.15 | 59.77 | 32.84 |
| DeiT-Ti INT4 | 3.8 | 70.43 | 57.87 | 31.53 |

Top-1 en %. ImageNet-R se evalúa sobre sus 200 clases, como en Hendrycks et al. (2021). DeiT-S en INT4 ocupa 2.2 MB más que DeiT-Ti en FP16 y acierta 7 puntos más en ImageNet-1k val, por eso DeiT-Ti aparece en todas las curvas.

## 3. El primer método publicado, AdaLog

Cada método publicado se ejecuta con el código y la configuración de sus autores, para que el punto en el gráfico sea su método y no nuestra versión de él. Cambian tres cosas, iguales para todos los métodos. Las imágenes de calibración son las primeras N líneas de `splits/calib_seed0.txt`, una imagen de entrenamiento por clase barajada con semilla 0, donde N es el número que usan los autores. Las activaciones quedan en precisión completa, porque no cambian el tamaño del archivo. La precisión sale de un archivo real, construido con los códigos y la grilla del propio método, releído, verificado contra los pesos que usa el código del método y evaluado en los tres conjuntos de prueba.

AdaLog (Wu et al., 2024) es el primer método ejecutado así, con sus 32 imágenes de calibración y sin reconstrucción por bloques. Su cuantizador logarítmico actúa sobre las activaciones, así que aquí se mide su cuantizador de pesos y la búsqueda que fija la grilla de cada fila. Cuantiza las 48 capas lineales de los bloques, la entrada de parches y la cabeza, con una grilla por fila de salida.

| AdaLog sobre DeiT-S | Archivo (MB) | ImageNet-1k val | ImageNet-V2 | ImageNet-R |
|---|---|---|---|---|
| W6 | 17.0 | 79.76 | 68.46 | 42.23 |
| W4 | 11.5 | 78.87 | 66.71 | 40.62 |
| W3 | 8.8 | 74.18 | 61.56 | 37.53 |

En los tres archivos, los pesos releídos son idénticos a los de AdaLog, y la precisión en ImageNet-1k val desde el archivo coincide con la que reporta el propio código de AdaLog. Las diferencias pareadas contra FP32 y sus intervalos están en `outputs/published_summary.json`.

## Código de terceros

Los métodos publicados no forman parte de este repositorio, porque cada uno tiene su propia licencia. `bash scripts/setup_third_party.sh` clona cada uno junto a esta carpeta, en `../third_party/`, en el commit que usamos, y aplica nuestro parche de `methods/<nombre>/` donde lo hay, con los nombres de carpeta que buscan `methods/*/run.sh` y `deitc/pruning.py`. VBP está bajo AGPL-3.0, así que nuestro código solo lo llama desde esa carpeta y no se copia nada de él aquí; revisa la licencia de cada repositorio antes de redistribuir algo.

| Método | Repositorio | Commit |
|---|---|---|
| AdaLog | https://github.com/GoatWu/AdaLog | 75a57d2 |
| APHQ-ViT | https://github.com/GoatWu/APHQ-ViT | 84c9fde |
| FIMA-Q | https://github.com/ShiheWang/FIMA-Q | 64c4b00 |
| RepQ-ViT | https://github.com/zkkli/RepQ-ViT | a160561 |
| PTQ4ViT | https://github.com/hahnyuan/PTQ4ViT | b42c790 |
| GRAIL | https://github.com/TWWinde/GRAIL_Compensation | ba5e48b |
| VBP (solo la etapa de poda) | https://github.com/boschresearch/variance-based-pruning | b6834b4 |
| GemLite (kernels para una comprobación) | https://github.com/dropbox/gemlite | f649e80 |

CORP no tiene código público; `deitc/pruning.py` es nuestra reimplementación a partir de su artículo.

## Cómo ejecutarlo

```bash
# Python 3.10 y una GPU con CUDA; usamos torch 2.11 y timm 1.0.30
pip install torch torchvision timm torchao safetensors scipy matplotlib jupyter huggingface_hub pyarrow
```

Los métodos publicados necesitan versiones más antiguas, así que corren en un segundo entorno con Python 3.10, torch 2.2.2 y timm 0.9.2, más lo que pide cada repositorio. Su intérprete va en `ADALOG_PY`, y `PY` apunta al primer entorno para los pasos en que el script de un método llama al nuestro.

ImageNet-V2 matched-frequency es público ([Hugging Face](https://huggingface.co/datasets/vaishaal/ImageNetV2)) e ImageNet-R está disponible por sus autores. ImageNet-1k requiere aceptar sus términos en Hugging Face; después, `python scripts/data/prepare_imagenet.py --out ../data/imagenet` escribe las 50 000 imágenes de validación y un conjunto de imágenes de entrenamiento para calibrar. Los cuadernos leen los tres conjuntos desde `../data/` o desde `IMAGENET_VAL_DIR`, `IMAGENETV2_DIR` e `IMAGENETR_DIR`.

```bash
jupyter nbconvert --to notebook --execute 01_rebuild_deit.ipynb    # cerca de un minuto en una RTX 4070
jupyter nbconvert --to notebook --execute 02_dense_baselines.ipynb
python scripts/data/make_calib_list.py --train ../data/imagenet/train --seed 0 # la lista de calibración común
bash scripts/setup_third_party.sh                                 # el código de los métodos publicados
```

Los cuadernos 03 a 06 leen los resultados de `outputs/`, así que muestran todas las tablas y gráficos desde un clon recién descargado sin ejecutar nada en la GPU. Con `RERUN = True` al inicio de uno, llama en cambio a los scripts que produjeron esos resultados, lo que toma horas, porque cada búsqueda de FIMA-Q o APHQ-ViT dura de 40 minutos a 2 horas en nuestra RTX 4070 de 12 GB. En esa tarjeta dos búsquedas a la vez no terminan antes que una tras otra, así que conviene ejecutarlas de una en una, lo que además mantiene válido cada tiempo de compresión. La latencia del cuaderno 06 se midió en esa tarjeta sin nada más corriendo, y los kernels de pocos bits que probamos no estaban compilados para ella, así que esos tiempos describen solo esta GPU. Todos los scripts se ejecutan desde la raíz del repositorio.

## Hoja de ruta

- [x] Reconstruir DeiT-S, verificarlo y contabilizar su tamaño
- [x] Las líneas base densas como archivos reales en ImageNet-1k val, V2 y R
- [x] La lista de calibración común y el flujo para métodos publicados, con AdaLog como el primero
- [x] Los demás métodos publicados con el mismo protocolo: APHQ-ViT, FIMA-Q, RepQ-ViT y PTQ4ViT, tal como se publicaron y solo pesos (FQ-ViT, I&S-ViT y ERQ quedan fuera)
- [x] Poda sin reentrenar: CORP, reimplementado con su compensación por mínimos cuadrados, GRAIL y la etapa de poda de VBP
- [x] Sensibilidad de cada capa y bloque a la poda y a los bits de los pesos
- [x] Asignación por bloque y las curvas de solo bits, solo poda y combinada
- [x] La serie W8A8 y las activaciones de 8 bits sobre nuestros modelos
- [ ] Qué activaciones de FIMA-Q son frágiles (ejecuciones terminadas, resultados en revisión)
- [ ] Redacción de los resultados en esta página y en el artículo

## Referencias principales

- Blalock, D., Gonzalez Ortiz, J. J., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning? *MLSys*.
- Hendrycks, D., et al. (2021). The many faces of robustness: A critical analysis of out-of-distribution generalization. *ICCV*.
- Li, Z., Xiao, J., Yang, L., & Gu, Q. (2023). RepQ-ViT: Scale reparameterization for post-training quantization of vision transformers. *ICCV*.
- Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet classifiers generalize to ImageNet? *ICML*.
- Touvron, H., et al. (2021). Training data-efficient image transformers & distillation through attention. *ICML*.
- Wu, Z., Chen, J., Zhong, H., Huang, D., & Wang, Y. (2024). AdaLog: Post-training quantization for vision transformers with adaptive logarithm quantizer. *ECCV*.
- Wu, Z., Wang, S., Zhang, J., Chen, J., & Wang, Y. (2025). FIMA-Q: Post-training quantization for vision transformers by Fisher information matrix approximation. *CVPR*.
