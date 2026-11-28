<div align="center" style="text-align: center;">

# Para-RAID-needle-LoRA

###### Русскоязычное LoRA-дообучение модели Needle 3 для управления SPIDOR через RaidAPI с использованием фильтрации и предварительной классификации на базе rubert-tiny2
###### ФАКТ МФТИ

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Needle_3-10B981?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0yMS4yIDIuOGEyLjggMi44IDAgMCAwLTQgMEwzLjUgMTguNmwtLjggMy4xYS41LjUgMCAwIDAgLjYuNmwzLjEtLjhMMjEuMiA2LjhhMi44IDIuOCAwIDAgMCAwLTR6bS0yLjMgMS40YTEgMSAwIDAgMSAxLjQgMS40bC0yLjEgMi4xLTEuNC0xLjQgMi4xLTIuMXoiLz48L3N2Zz4%3D" alt="Needle 3">
<img src="https://img.shields.io/badge/rubert--tiny2-FFD21E?style=for-the-badge" alt="rubert-tiny2">

</div>

---

Пайплайн системы Text2RaidAPI:
- Слой 1 (фильтр и классификатор): `rubert-tiny2` + `offtopic_clf.joblib`
- Слой 2 (NLU-движок): `needle3` + `LoRA`

---

## Базовые модели и авторы

> **Needle 3.0** (*«Needle: Automation Foundation Model for Tiny Devices»*, Cactus Compute, Inc., 2026)
> **Авторы:** Henry Ndubuaku, Karen Mosoyan, Jakub Mroz, Noah Cylich, Satyajit Kumar, Parkirat Sandhu, Roman Shemet, Justin H. Lee — [cactus-compute/needle](https://github.com/cactus-compute/needle)

> **rubert-tiny2** (*«cointegrated/rubert-tiny2»*, 2022)
> **Автор:** David Dale — [cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2)
