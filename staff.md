---
layout: page
title: Nastavnici
description: Pregled svih nastavnika na predmetu.
---

# Nastavnici
## Predavanja

{% assign instructors = site.staffers | where: 'lectures', true %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

{% assign teaching_assistants = site.staffers | where: 'labs', true %}
{% assign num_teaching_assistants = teaching_assistants | size %}
{% if num_teaching_assistants != 0 %}
## Laboratorijske vježbe

{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}
{% endif %}
