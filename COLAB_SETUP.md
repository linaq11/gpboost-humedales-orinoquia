# Cómo abrir los notebooks en Google Colab

Hay tres rutas. Elige la que mejor le sirva a Santiago:

---

## Opción A — GitHub público + URL directa (recomendada)

Esta es la forma más limpia: subes el repo a GitHub y cualquiera lo abre en Colab cambiando `github.com` por `colab.research.google.com/github`.

### Pasos

1. Crear repo en GitHub (web):
   - Nombre: `examen-2-gpboost`
   - Público (o privado + invitación a Santiago)
   - **No** inicialices con README ni .gitignore (ya los tenemos en local)

2. Subir el contenido desde la carpeta local:
   ```bash
   cd "C:\LINA\UNAL 2626-1\PERCEPCION REMOTA\Examen_2_GPBoost"
   git init
   git add .
   git commit -m "Examen 2 GPBoost — entrega inicial"
   git branch -M main
   git remote add origin https://github.com/<TU_USUARIO>/examen-2-gpboost.git
   git push -u origin main
   ```

3. URL para abrir cualquier notebook en Colab:
   ```
   https://colab.research.google.com/github/<TU_USUARIO>/examen-2-gpboost/blob/main/<NOTEBOOK>.ipynb
   ```
   Ejemplo con el notebook principal:
   ```
   https://colab.research.google.com/github/<TU_USUARIO>/examen-2-gpboost/blob/main/Examen_2_GPBoost.ipynb
   ```

4. Pega esa URL en el README como badge:
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<TU_USUARIO>/examen-2-gpboost/blob/main/Examen_2_GPBoost.ipynb)
   ```

### Lo que el notebook necesita en la primera celda de Colab

En Colab no hay `gpboost` ni `shap` preinstalados. Agrega esta celda al inicio (o
deja una versión `*_colab.ipynb` con la celda incluida):

```python
# Solo si se ejecuta en Google Colab
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    !pip -q install gpboost shap

    # Montar datos. Hay dos rutas:
    # (a) Clonar el repo dentro del runtime:
    !git clone -q https://github.com/<TU_USUARIO>/examen-2-gpboost.git /content/repo
    %cd /content/repo

    # (b) Alternativa: montar Google Drive si Santiago guardó allí los CSV grandes
    # from google.colab import drive
    # drive.mount('/content/drive')
```

---

## Opción B — Compartir el .ipynb por Google Drive

Más rápido si no quieres usar GitHub todavía. Limitación: cada uno tiene una copia
independiente que se desincroniza al editar.

1. Sube el `.ipynb` a una carpeta compartida de Drive (compartir con
   `srodriguezba@unal.edu.co` o el correo de Santiago).
2. Santiago hace clic derecho → **Abrir con → Google Colaboratory**.
3. Si edita y quiere guardar cambios, Colab le ofrece "Guardar una copia en Drive"
   (queda separada de la tuya) o "File → Save" sobre la misma copia.
4. Para que el notebook encuentre los CSV, sube también la carpeta `data/` al mismo
   Drive y al inicio del notebook agrega:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   BASE = '/content/drive/MyDrive/<carpeta_compartida>/Examen_2_GPBoost'
   ```

---

## Opción C — Colab + GitHub Gist privado

Para iteración rápida sin crear repo. Subes el `.ipynb` como
[Gist](https://gist.github.com) y compartes el enlace. Colab lo abre con la URL
`https://colab.research.google.com/gist/<gist_id>`. No soporta `data/` automáticamente
— Santiago tendría que descargar los CSV manualmente.

---

## Checklist antes de pasar el repo a Santiago

- [ ] `requirements.txt` contiene versiones fijas (ya está)
- [ ] Primera celda detecta Colab e instala `gpboost`, `shap` si hace falta
- [ ] Las rutas dentro del notebook son **relativas** al repo, no absolutas a `C:\LINA\...`
- [ ] `data/`, `modelos/`, `figuras/` están en el repo (chequear `.gitignore`)
- [ ] README incluye el badge "Open in Colab" con la URL correcta
- [ ] Probado abriendo el notebook en Colab y ejecutando "Runtime → Run all"
- [ ] Santiago tiene cuenta de Google y acepta cuotas de GPU/CPU de Colab Free

---

## Notas operativas para Lina

- **Tamaños:** los `.ipynb` con figuras embebidas pueden pasar de 5–10 MB. GitHub los
  acepta sin problema (límite 100 MB por archivo).
- **`.gitignore`:** verifica que NO ignore `data/*.csv` ni `modelos/*.json` ni
  `figuras/*.png` (Santiago necesita acceso a los outputs derivados; los datos
  primarios externos sí van ignorados).
- **Reproducibilidad:** las semillas (`np.random.seed(1)`) y `requirements.txt` con
  pins exactos son suficientes — Santiago debería reproducir los mismos números.
- **Colab GPU:** GPBoost no necesita GPU. Usa Runtime → Change runtime type → CPU,
  ahorra tiempo de arranque.

---

## URL plantilla rápida (cuando ya tengas el repo)

Reemplaza `linaq11` por tu usuario real de GitHub:

| Notebook | Open in Colab |
|---|---|
| Notebook único | `https://colab.research.google.com/github/linaq11/examen-2-gpboost/blob/main/Examen_2_GPBoost.ipynb` |
| Notebook 01 (si los separas) | `https://colab.research.google.com/github/linaq11/examen-2-gpboost/blob/main/notebooks/01_reproduccion_simulacion.ipynb` |
