---
layout: page
title: About Us
description: A listing of all team members and mentor.
---

{: .fs-7 .fw-400 }

## Mentor

{% assign instructors = site.people | where: 'role', 'Mentor' %}
{% for people in instructors %}
{{ people }}
{% endfor %}

## Students

{% assign students = site.people | where: 'role', 'Student' %}
<div class="role" style="display: flex; flex-wrap: wrap; margin-bottom: 10px;">
    {% for people in students %}
    {{ people }}
    {% endfor %}
</div>
