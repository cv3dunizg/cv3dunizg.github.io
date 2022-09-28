---
layout: page
title: Prva laboratorijska vježba
description: Upute za prvu laboratorijsku vježbu.
nav_exclude: true
---

# Prva laboratorijska vježba: geometrijske deformacije slika

U računalnom vidu često trebamo modificirati slike 
različitim geometrijskim deformacijama.
Ova laboratorijska vježba razmatra `unatražne` deformacije
koje se najčešće koriste u praksi.
Označimo ulaznu sliku s $I_s$, 
izlaznu sliku s $I_d$,
te parametriziranu koordinatnu 
transformaciju s $\mathbf{T}_p$.
Tada unatražnu deformaciju slike 
možemo formulirati sljedećim izrazom:
$$I_d (\mathbf{q}) = I_s (\mathbf{T}_p(\mathbf{q})) \quad.$$

Kako slike tipično imaju dvije geometrijske osi,
i domena i kodomena koordinatnih transformacija 
odgovarat će Euklidskoj ravnini:
$\mathbf{T}_p : \mathbb{R}^2 \rightarrow \mathbb{R}^2$.
U praksi se koriste različite 
vrste geometrijskih transformacija,
ali najčešće koristimo 
afine, projekcijske i radijalne transformacije.
Afine i projekcijske transformacije 
čuvaju kolinearnost točaka,
dok radijalne transformacije ne utječu na 
udaljenost od ishodišta koordinatnog sustava.

## Afine transformacije

Afine transformacije ravnine možemo prikazati sljedećom jednadžbom:
$$\mathbf{q}_d = \mathbf{T}_p(\mathbf{q}_s)) = \mathbf{A} \cdot \mathbf{q}_s + \mathbf{b} \quad.$$
Tablica prikazuje hijerarhijski popis 
vrsta afinih transformacija 
tako da svaka sljedeća vrsta 
odgovara poopćenju prethodne: 

| *transformacija*                | *stupnjevi slobode* | *invarijante* | *ograničenja* |
| ----------------                | ------------------- | ------------- | --------- |
| translacija                     |           2         | orijentacija  | $\mathbf{A}=\mathbf{I}$ |
| transformacija krutog tijela    |           3         | udaljenost    | $\mathbf{A}=\mathbf{R}$, $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$|
| sličnost                        |           4         | kutovi        | $\mathbf{A}=s\mathbf{R}$, $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$|
| općenita afina transformacija   |           6         | paralelnost   | nema |

## Bilinearna interpolacija

