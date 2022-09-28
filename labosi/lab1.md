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
$$I_d (\mathbf{q}) = I_s (\mathbf{T}_p(\mathbf{q})) \ .$$

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
$$\mathbf{q}_d = \mathbf{T}_p(\mathbf{q}_s) = \mathbf{A} \cdot \mathbf{q}_s + \mathbf{b} \ .$$
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

Ako se odnos između dvije slike
može opisati afinom deformacijom,
onda parametre koordinatne transformacije
možemo izlučiti iz korespondencija.
Neka su u izvorišnoj slici 
$I_s$
zadane točke
$\mathbf{q}_{si}$. [//]: <> (_)
Neka su u odredišnoj slici 
$I_d$ 
zadane korespondentne točke 
$\mathbf{q}_{di}$.
Tada za svaki par korespondencija vrijedi:

$$\eqalign{
a_{11} q_{si1} + a_{12} q_{si2} + b_1 &= q_{di1}\\  
a_{21} q_{si1} + a_{22} q_{si2} + b_2 &= q_{di2}}$$

Ovu jednadžbu možemo prikazati
kao sustav od dvije jednadžbe 
sa šest nepoznanica:

$$ {\left\lbrack \matrix{q_{si1} & q_{si2} & 0 & 0 & 1 & 0\cr 0 & 0 & q_{si1} & q_{si2} & 0 & 1} \right\rbrack} 
\cdot \left\lbrack \matrix{a_{11} \cr a_{12} \cr a_{21} \cr a_{22} \cr b_{1} \cr b_{2}} \right\rbrack
= \left\lbrack \matrix{q_{di1} \cr q_{di2}} \right\rbrack
$$

Ako dodamo još dvije korespondencije, dobit ćemo sustav $6\times 6$
koji ima jedinstveno rješenje.
Tražena deformacija biti će određena rješenjem tog sustava.

## Projekcijske transformacije

Projekcijske transformacije ravnine 
možemo prikazati sljedećom jednadžbom
(primijetite da je brojnik vektor, a nazivnik skalar):
$$\mathbf{q}_d = \mathbf{T}_p(\mathbf{q}_s) = \frac{\mathbf{A} \cdot \mathbf{q}_s + \mathbf{b}}{\mathbf{w}^\top\mathbf{x} + w_0} \ .$$


## Bilinearna interpolacija

