---
layout: page
title: Nastavnici
description: Pregled svih nastavnika na predmetu.
---

# Nastavnici
## Predavanja

{% assign instructors = site.staffers | where: 'role', 'Instructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

{% assign teaching_assistants = site.staffers | where: 'role', 'Teaching Assistant' %}
{% assign num_teaching_assistants = teaching_assistants | size %}
{% if num_teaching_assistants != 0 %}
## Laboratorijske vježbe

{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}
{% endif %}
