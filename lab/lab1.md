---
layout: page
title: Prva laboratorijska vježba
description: Upute za prvu laboratorijsku vježbu.
nav_exclude: true
---


# Prva laboratorijska vježba: geometrija dvaju pogleda

Mnoge metode računalnog vida razmatraju scenarij 
gdje je na temelju više pogleda na scenu 
potrebno izlučiti relativnu orijentaciju kamera
i rekonstruirati trodimenzionalnu strukturu scene.
Sasvim općenito, možemo reći da te metode
razmatraju rekonstrukciju geometrije više pogleda.
Vrlo važan specijalni slučaj razmatra jednu kameru 
koja se giba kroz scenu koja se sastoji 
od pokretnih objekata i nepokretne pozadine. 
U tom slučaju možemo pričati o izlučivanju 
strukture iz kretanja (eng. structure from motion).

Metode 3D rekonstrukcije možemo podijeliti 
na kalibrirani i nekalibrirani slučaj.
Kalibrirani slučaj imamo kada piksele
možemo jednoznačno preslikati 
na jediničnu sferu sa središtem u žarištu kamere.
Nekalibrirani slučaj imamo kada naš program
ne zna s kojom kamerom je pribavljena slika.
Radi jednostavnosti, u ovoj vježbi razmatrat ćemo
samo kalibrirani slučaj koji je i najvažniji u praksi.

Sve metode 3D rekonstrukcije u prvom koraku
izlučuju relativnu orijentaciju (eng. relative orientation) dvaju kamera
pa ćemo upravo taj zadatak razmatrati i u ovoj vježbi.
Zadatak ćemo rješavati pod pretpostavkom 
da je koordinatni sustav svijeta poravnat 
s koordinatnim sustavom prve kamere.
Ta pretpostavka neće smanjiti općenitost rješenja.
Stoga, projekcijska matrica prve kamere biti će 
$$\mathbf{P}_1 = \left\lbrack \mathbf{I} | \mathbf{0} \right\rbrack$$,
dok će naš postupak trebati izlučiti 
projekcijsku matricu druge kamere 
$$\mathbf{P}_2 = \left\lbrack \mathbf{R} | \mathbf{t} \right\rbrack$$.

Valja napomenuti da će svi naši postupci moći izlučiti 
samo smjer translacije ali ne i njen iznos.
Razlog tome je nemogućnost razlikovanja 
iznosa translacije od globalnog mjerila 
trodimenzionalne rekonstrukcije scene.
Drugim riječima, ako umjesto prave scene
promatramo s$$\times$$ umanjenu maketu
te s jednakim faktorom s umanjimo 
i pomak druge kamere -
promatrane slike neće se promijeniti. 

## Sintetički eksperimentalni postav
[source](src/ create_2vg_setup.cxx)

## Algoritam s osam točaka

## Dekompozicija esencijalne matrice

## Poboljšanja osnovnog postupka

### Normalizacija koordinata

### Robusna estimacija (bonus)

## Rekonstrukcija 3D strukture (bonus)

