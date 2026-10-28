# Decisiones

*[English version](DECISIONS.md)*

Estas son las decisiones tomadas a lo largo del proyecto, con una breve razón para cada una.

| Fecha | Decisión | Por qué |
|---|---|---|
| 2026-10-23 | Estudiar DeiT-Small, con DeiT-Tiny como la alternativa más pequeña sin comprimir. | Entender el modelo que se va a comprimir y tener algo con qué compararlo. |
| 2026-10-23 | Cada punto se evalúa en ImageNet-1k val, ImageNet-V2 e ImageNet-R, con intervalos bootstrap pareados sobre las mismas imágenes. | Un solo conjunto de prueba puede ocultar cómo se comporta la compresión ante un cambio de distribución, y las diferencias pequeñas necesitan una prueba pareada. |
| 2026-10-23 | Solo se replican los métodos que no requieren entrenamiento y tienen código público. | Podemos ejecutarlos con un mismo protocolo; los que reentrenan se listan, pero quedan fuera porque no tenemos el cómputo ni el kernel especializado. |
| 2026-10-23 | Los métodos de reducción de tokens quedan fuera del gráfico. | No cambian el tamaño del archivo. |
| 2026-10-23 | La grilla de pasos de poda y de bits de nuestro propio método aún no está fija. | Se fijará después de un piloto que mida la sensibilidad de cada bloque. |
| 2026-10-26 | Solo se comprimen los pesos; las activaciones quedan en precisión completa. | Las activaciones no cambian el tamaño del archivo. |
| 2026-10-26 | Cada método publicado se ejecuta con el código y la configuración de sus autores: las capas que cuantiza, la forma de su grilla y su número de imágenes de calibración. | El punto en el gráfico debe estar lo más cerca posible del método de los autores para que la comparación sea precisa. |
| 2026-10-26 | Solo cambian tres cosas para todos los métodos: las imágenes de calibración, los conjuntos de prueba y las activaciones. | Queremos probar los modelos en las mismas condiciones. |
| 2026-10-26 | Las imágenes de calibración son una imagen de entrenamiento por clase, barajadas con semilla 0 (`splits/calib_seed0.txt`), y un método que usa N imágenes toma las primeras N. | Todos los métodos usan el mismo conjunto de imágenes, barajado de la misma forma, para representar tantas clases como sea posible. |
| 2026-10-26 | La salida de cada método se guarda como un archivo empaquetado real, con los valores de su grilla en la precisión que usa el método, y los pesos releídos deben ser iguales bit a bit a los del propio método. | El código publicado guarda los pesos en FP32 y los redondea durante la pasada hacia adelante, así que su checkpoint no muestra el tamaño real. |
| 2026-10-26 | El redondeo simple se mantiene en el gráfico como referencia. | Muestra cuánto gana la búsqueda de cada método en cada tamaño. |
| 2026-10-26 | La latencia queda fuera por ahora. | Los archivos empaquetados se expanden a flotantes para ejecutarse, así que el tiempo no reflejaría los formatos comprimidos, y no tenemos el kernel necesario para inferencia en estos tamaños. |
