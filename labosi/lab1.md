---
layout: page
title: Prva laboratorijska vježba
description: Upute za prvu laboratorijsku vježbu.
nav_exclude: true
---


# Prva vježba: geometrijske deformacije slika

U računalnom vidu često trebamo modificirati slike 
različitim geometrijskim deformacijama
kao što su uvećanje, rotiranje ili izrezivanje.
Ova laboratorijska vježba razmatra _unatražne_ deformacije
koje se najčešće koriste u praksi.
Označimo ulaznu sliku s $$I_s$$, 
izlaznu sliku s $$I_d$$,
vektor cjelobrojinh pikselskih koordinata s $$\mathbf{q}$$,
te parametriziranu koordinatnu 
transformaciju s $$\mathbf{T}_p$$.
Tada unatražnu deformaciju slike 
možemo formulirati sljedećim izrazom: 

$$I_d (\mathbf{q}) = I_s (\mathbf{T}_p(\mathbf{q})) \ .$$

Kako slike tipično imaju dvije geometrijske osi,
i domena i kodomena koordinatnih transformacija 
odgovarat će Euklidskoj ravnini: 
$$\mathbf{T}_p : \mathbb{R}^2 \rightarrow \mathbb{R}^2$$.
U praksi se koriste različite 
vrste geometrijskih transformacija,
ali najčešće koristimo 
afine, projekcijske i radijalne transformacije.
Afine i projekcijske transformacije 
čuvaju kolinearnost točaka,
dok radijalne transformacije ne utječu na 
udaljenost od ishodišta koordinatnog sustava.

## Afine transformacije

Označimo početni 2D vektor realnih koordinata s $$\mathbf{q}_s$$,
konačni 2D vektor realnih koordinata s $$\mathbf{q}_d$$, 
linearno ravninsko preslikavanje s $$A$$,    
te 2D pomak s $$b$$.
Tada afinu transformacijeu ravnine 
možemo prikazati sljedećom jednadžbom: 

$$\mathbf{q}_d = \mathbf{T}_p(\mathbf{q}_s) = \mathbf{A} \cdot \mathbf{q}_s + \mathbf{b} \ .$$

Tablica prikazuje hijerarhijski popis 
vrsta afinih transformacija 
tako da svaka sljedeća vrsta 
odgovara poopćenju prethodne
($$\mathbf{I}$$=$$\mathrm{diag}(1,1),
 $$\mathbf{R}$$ je matrica rotacije 2D podataka): 

| *transformacija*                | *stupnjevi slobode* | *invarijante* | *ograničenja* |
| ----------------                | ------------------- | ------------- | --------- |
| translacija                     |           2         | orijentacija  | $$\mathbf{A}=\mathbf{I}$$ |
| transformacija krutog tijela    |           3         | udaljenost    | $$\mathbf{A}=\mathbf{R}$$, $$\mathbf{R}^\top\mathbf{R}=\mathbf{I}$$ |
| sličnost                        |           4         | kutovi        | $$\mathbf{A}=s\mathbf{R}$$, $$\mathbf{R}^\top\mathbf{R}=\mathbf{I}$$ |
| općenita afina transformacija   |           6         | paralelnost   | nema |

Ako se odnos između dvije slike
može opisati afinom deformacijom,
onda parametre koordinatne transformacije
možemo izlučiti iz korespondencija.
Neka su u izvorišnoj slici 
$$I_s$$
zadane točke
$$\mathbf{q}$$<sub>si</sub>.
Neka su u odredišnoj slici 
$$I_d$$
zadane korespondentne točke 
$$\mathbf{q}_{di}$$.
Tada za svaki par korespondencija vrijedi:

$$\eqalign{
a_{11} q_{si1} + a_{12} q_{si2} + b_1 &= q_{di1}\\  
a_{21} q_{si1} + a_{22} q_{si2} + b_2 &= q_{di2}}$$

Ove dvije jednadžbe možemo presložiti
tako da 6 parametara afine transformacije 
istaknemo kao nepoznanice
te ih zapisati u matričnom obliku kako slijedi:

$$ {\left\lbrack \matrix{q_{si1} & q_{si2} & 0 & 0 & 1 & 0\cr 0 & 0 & q_{si1} & q_{si2} & 0 & 1} \right\rbrack} 
\cdot \left\lbrack \matrix{a_{11} \cr a_{12} \cr a_{21} \cr a_{22} \cr b_{1} \cr b_{2}} \right\rbrack
= \left\lbrack \matrix{q_{di1} \cr q_{di2}} \right\rbrack
$$

Ako dodamo još dvije korespondencije, dobit ćemo sustav $$6\times 6$$
koji ima jedinstveno rješenje, osim ako su korespondencije kolinearne.
Tražena deformacija biti će određena rješenjem tog sustava.

## Projekcijske transformacije

Projekcijske transformacije ravnine 
možemo prikazati sljedećom jednadžbom
(primijetite da je brojnik vektor, a nazivnik skalar): 

$$\mathbf{q}_d = \mathbf{T}_p(\mathbf{q}_s) = \frac{\mathbf{A} \cdot \mathbf{q}_s + \mathbf{b}}{\mathbf{w}^\top\mathbf{x} + w_0} \ .$$

Projekcijske transformacije možemo odrediti iz korespondencija
na vrlo sličan način kao što smo to pokazali za afine transformacije.
Međutim, lako se vidi da su u ovom slučaju nepoznanice određene
do proizvoljne multiplikativne konstante koja se u razlomku pokrati. 
Ako prikupimo četiri korespondencije, dobit ćemo 
homogeni sustav s devet nepoznanica oblika 
$$\mathbf{M}\mathbf{x}=\mathbf{0}$$.
Pokazuje se da takav sustav 
ima točno jedno netrivijalno rješenje
ako niti jedna trojka korespondencija nije kolinearna.
Rješenje odgovara 
[desnom singularnom vektoru](https://en.wikipedia.org/wiki/Singular_value_decomposition#Solving_homogeneous_linear_equations)
matrice $$\mathbf{M}$$
koji odgovara singularnoj vrijednosti nula.
Ako imamo višak ograničenja (više od 4 korespondencije)
optimalno rješenje u algebarskom smislu
dobivamo kao desni singuarni vektor
koji odgovara najmanjoj singularnoj vrijednosti matrice $$\mathbf{M}$$.

## Interpolacija piksela

Ranije smo najavili da unatražnu deformaciju slike 
formuliramo sljedećim izrazom: 

$$I_d (\mathbf{q}) = I_s (\mathbf{T}_p(\mathbf{q})) \ .$$

Primijetimo da koordinatna transformacija nije diskretna,
tj. da 2D vektor $$\mathbf{T}_p(\mathbf{q})$$ ima realne koordinate.
To znači da u odredišni piksel $$\mathbf{q}$$ valja upisati
piksel koji se nalazi "između" piksela izvorne slike.
Ovaj problem nazivamo interpolacijom slike.
Postoji više načina da se to napravi,
a ovdje ćemo upoznati interpolaciju najbližim susjedom
i bilinearnu interpolaciju.

### Interpolacija najbližim susjedom


### Bilinearna interpolacija

Bilinearna interpolacija
[literatura](http://www.zemris.fer.hr/~ssegvic/project/pubs/bosilj10bs.pdf)

## Zadatak 1: interpoliranje

Napišite kod koji učitava sliku,
te na nju primjenjuje slučajnu afinu transformaciju
primjenom i) interpolacije najbližim susjedom
te ii) bilinearnom interpolacijom. 
Upute:
- poslužite se slikama `scipy.misc.ascent()` i `scipy.misc.face()`
- matricu $$\mathbf{A}$$ slučajne afine transformacije zadajte ovako: `A = .25*np.eye(2) + np.random.normal([2,2])`
- vektor $$\mathbf{b}$$ slučajne afine transformacije zadajte tako da se središnji piksel izvorišne slike preslika u središnji piksel odredišne slike
- napišite funkciju `affine_nn(Is, A,b, Hd,Wd)` koja izvorišnu sliku `Is` deformira u skladu s parametrima `A` i `b` te odredišnu sliku rezolucije `Hd`$$\times$$`Wd` vraća u povratnoj vrijednosti; odredišni pikseli koji padaju izvan izvorišne slike trebaju biti crni; funkcija treba koristiti interpolaciju najbližim susjedom te funkcionirati i za sive slike i za slike u boji
- napišite funkciju `affine_bilin(Is, A,b, Hd,Wd)` koja radi isto što i `affine_nn`, ali s bilinearnom interpolacijom 
- neka odredišna rezolucija bude `Hd`$$\times$$`Wd` = 200$$\times$$200
- ispišite standardnu devijaciju odstupanja odgovarajućih piksela u dvije slike
- neka vaš glavni program odgovara sljedećem kodu: 

```
import matplotlib.pyplot as plt
import scipy.misc as misc
import numpy as np

Is = misc.face()
Is = np.asarray(Is)

Hd,Wd = 200,200
A,b = recover_affine_diamond(Is.shape[0],Is.shape[1], Hd,Wd)

Id1 = affine_nn(Is, A,b, Hd,Wd)
Id2 = affine_bilin(Is, A,b, Hd, Wd)
# dodati ispis standardne devijacije

fig = plt.figure()
if len(Is.shape)==2: plt.gray()
for i,im in enumerate([Is, Id1, Id2]):
  fig.add_subplot(1,3, i+1)
  plt.imshow(im.astype(int))
plt.show()
```

## Zadatak 2: određivanje parametara afine transformacije iz korespondencija

Napišite funkciju `recover_affine_diamond(Hs,Ws, Hd,Wd)` koja vraća parametre afine transformacije
koja piksele _središta stranica_ izvorišne slike dimenzija Hs$$\times$$Hs 
preslikava u _kuteve_ odredišne slike dimenzija Hd$$\times$$Hd . 
Upute:
- za rješavanje sustava jednadžbi koristite `np.linalg.solve`

## Zadatak 3: određivanje parametara projekcijske transformacije iz korespondencija

Napišite funkciju `recover_projective(Qs, Qd)` koja vraća parametre projekcijske transformacije
ako su zadane točke izvorišne slike `Qs` i točke odredišne slike `Qd`. 
Upute:
- za rješavanje homogenog sustava koristite `np.linalg.svd`
