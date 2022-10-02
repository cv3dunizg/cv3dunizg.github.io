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
promatramo $$s\times$$ umanjenu maketu
te s jednakim faktorom $$s$$ umanjimo 
i pomak druge kamere -
dobit ćemo iste slike kao i u originalnom slučaju. 

## Sintetički eksperimentalni postav

Metode relativne orijentacije kamera 
teško je evaluirati na stvarnim slikama
zbog teškog mjerenja stvarnih pomaka.
Zbog toga ćemo ovu vježbu provoditi
na sintetičkom eksperimentalnom postavu
gdje dvije kamere promatraju 
slučajni oblak točaka.
Radi jednostavnosti, obje kamere 
nalaze se u ravnini x-z referentne kamere,
a udaljenost među ishodištima kamera
uvijek ima jediničnu normu.
Stoga pomak druge kamere nedvosmisleno možemo opisati
pomakom smjera gledanja druge kamere 
te orijentacijom spojnice dvaju ishodišta.
Oblak točaka instanciramo u kvadru za kojeg zadajemo
udaljenost od referentne kamere i dubinu.
Stoga, eksperimentalni postav 
zadajemo sljedećim parametrima:
- $$\theta$$: smjer gledanja kamere C<sub>B</sub> u odnosu na referentnu kameru C<sub>A</sub> (stupnjevi)
- $$\phi$$: smjer pomaka kamere C<sub>B</sub> u odnosu na referentnu kameru C<sub>A</sub> (stupnjevi)
- $$D$$: udaljenost oblaka točaka,
- $$d$$: dubina oblaka točaka,
- $$\delta$$: nagib oblaka točaka,
- $$N$$: broj točaka.
Sljedeća slika ilustrira navedene parametre 
i prikazuje tri konkretne konfiguracije:

![Sintetički postav za vrednovanje postupaka za izlučivanje relativne orijentacije dvaju kamera](../assets/images/2vg_setup.png)

Slika ilustrira i kako se oblak točaka instancira
samo u dijelovima scene koji su vidljivi u obje kamere.
Da bismo to mogli provesti, potrebni su nam 
intrinsični parametri kamera
(pretpostavljamo da obje kamere imaju iste parametre).
Konačno, kako bismo izmjerili otpornost metode na šum,
svakoj projiciranoj točci dodajemo
slučajan normalni šum varijance $$\sigma$$.
Stoga valja zadati i sljedeće parametre eksperimentalnog postava:
- `\alpha_H`: horizontalno vidno polje u stupnjevima,
- `h`, `w`: dimenzije slike,
- $$\sigma$$: standardna devijacija šuma u pikselima.

Eksperimentalni postav možemo instancirati
primjenom sljedećeg 
[programa](../src/create_2vg_setup.cxx).
Program bi se trebao moći prevesti 
s bilo kojim standardnim prevoditeljem
(mi smo testirali g++ i MSVC).
Javite ako bude bilo bilo kakvih problema.
Parametri postava zadaju se u naredbenom retku. 
Evo primjera naredbenog retka koji 
instancira postav na lijevoj slici:
```
./create_2vg_setup -5_90_10_5_0_10000 45_384_288_100 >exp.data
```
Navedeni primjer zadaje $$\theta=-5^\circ$$,
$$\phi=90^\circ$$, $$D=10$$, $$d=5$$, $$\delta=0$$,
$$N=10000$$, $$\alpha_H$$=$$45^\circ$$,
`h`,`w`=384,288, $$\sigma=1.00$$.
Primijetite da zbog lakšeg parsanja,
program zahtijeva da sve parametre upišemo
kao cjelobrojne konstante te 
da se zadaje standardna devijacija pomnožena sa 100.
Ako za prvi argument zadamo xx, smjer gledanja druge kamere
odabrat će se tako da presjecište smjerova gledanja
bude u sredini oblaka točaka. Evo primjera:
```
./create_2vg_setup xx_00_10_5_0_10000 45_384_288_100 >exp.data
```

Program za kreiranje postava ispisuje 
projekcijsku matricu druge kamere
te dva polja po N projiciranih točaka 
za dvije kamere pretpostavljenog postava.
Pojedini elementi ispisa razdvojeni su praznim retkom.
U svim prikazanim primjerima, ispis programa 
preusmjerava se u datoteku `exp.data`, ali u praksi, 
ako program pokrećemo iz naredbenog retka,
možemo koristiti i ulančavanje procesa. 
Matricu kamere `P` te vektore točaka `qas` i `qbs`
iz Pythona možemo čitati sljedećim kodom:
```
import numpy as np
import itertools
import sys

def makegen(f):
  return ( np.array([float(c) 
    for c in line[1:-2].split(',')])
      for line in itertools.takewhile(lambda x: x != "\n", f))

# f = open('exp.data')
f = sys.stdin
line = f.readline().split('),(')
line[0] = line[0][7:]
line[2] = line[2][:-3]
P = np.array([ [float(x)  for x in row.split(',')] for row in line])
f.readline()

qas = np.array(list(makegen(f)))
qbs = np.array(list(makegen(f)))
```

U ovoj vježbi usredotočit ćemo se na sljedeće konfiguracije:
- deset različitih pomaka: $$\theta \in$$ `range(0,91,10)` (stupnjevi)
- konvergentni kutevi gledanja: $$\phi$$ = `'xx'`
- D,d,$$\delta$$ = 10,5,0
- N = 10000
- uobičajena širina vidnog polja: $$\alpha_H$$ = $$45^\circ$$ 
- h,w = 288, 384
- standardna devijacija pogreške: jedan piksel, $$\sigma$$ = 1.00

Ispitni program treba za svaki pomak provesti 
nexp=100 eksperimenata nad uzorkom 
od szSample=50 slučajnih parova točaka,
te zabilježiti kutnu po grešku 
rekonstruiranog vektora translacije.

## Algoritam s osam točaka

## Dekompozicija esencijalne matrice i procjena pogreške

## Poboljšanja osnovnog postupka

### Normalizacija koordinata

### Robusna estimacija (bonus)

## Rekonstrukcija 3D strukture (bonus)

