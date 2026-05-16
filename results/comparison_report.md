# Phase 8 Results and Comparison Report

This report summarizes training curves, quantitative evaluation, and sample generations from the trained LSTM and Transformer checkpoints.

## Quantitative Comparison

| Model | Final Train Loss | Final Val Loss | Test Loss | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| Lstm | 3.5611 | 3.2959 | 3.3454 | 28.3716 |
| Transformer | 3.4137 | 3.0302 | 3.1143 | 22.5187 |

Best model by perplexity: **Transformer**

## Loss Curve Summary

The saved plot in `results/loss_curves.png` compares training and validation loss for both models. In the smoke-training run, the Transformer converged to a lower validation loss than the LSTM.

## Qualitative Samples

### Lstm
- Temperature 0.5
  - Sample 1: `To be or not to be  i ao o      n    sas mf  n ee    n eo tsln   ns ye  e ee e  d tnilimutele \n i r e   i n , \n ulm   i en s  in   iP  w         aa  a i  iai e    no bsd e     an`
  - Sample 2: `To be or not to bel       i y o n    l td yo,  eri         pi  Iir ore ii   i rlc ra  i e    n w  rt iW wea in    ia  e o  a r id   ml n e   \n  s   nar    n  n     n\n  aa   i,nr `
  - Sample 3: `To be or not to bee ne ,,  s  o   rsdf        dln r ir   i  r,d t   e er     i el t\ni  u  n ew \n n    s    wwo \n rn nda    nt n i   \n W  ie ma dm  i  mtlenn   r  ntAio ui\n nn ...`
- Temperature 1.0
  - Sample 1: `To be or not to be siyaoeo,yfr sn  Nssapimfr neeehm  a eo tsln Y ns ye  f eene.ad hnllimugelep\n i grec  phnE, \n,ulm  vdkmnlss indo iPspw X aRw sdaagcani  sS;he \n- no bsd e '\n h...`
  - Sample 2: `To be or not to belno  I vify onc  dul tdpyo,Wceridv a sho pi,nIirlorekiiIeyimglc raQside o mb w nytrIW wda .nl osia rebos a roid\n: mlSn e  a\ni s b naru B'nbugi,  ru\noyg: s y,nr...`
  - Sample 3: `To be or not to bee\nne\n,,k s com hrsufb r  p lRlhzruIr  Eig,r,d m uie erc Ftii el t\ni  u\ntu ew \n nUua s tf wwo \nern ndaBy; nN nrkmo \n W uiekmafdm\ndI, mtl.nn,Tlr wntAYo uiII...`
- Temperature 1.5
  - Sample 1: `To be or not to be wiy:oeo,yfr snguNssapZmfp neeehm  a eostslnOY ns ye  f eene.-d hnllimugelep\n KegrJc mphnE, f,ulm: vdkmnlKsNinPo iPspw X ORwaWdaagcN.icFfS;he \n-\nnovbsdje '\nrh...`
  - Sample 2: `To be or not to belno  Iivify HFcD dul kdpyo,Wceridv a shovpi,nIarlorekiiIJyiUggcrr,QsiOe o mbmw nytQIW wda .nl oEia3rebos aWroid\n: mlSn eu a\ni sCb narusB'nbDgi,I'rB\n$yg: zty,Vr...`
  - Sample 3: `To be or not to bee\nne\n,,kks comeArsufbdr  p $RlhzruIr EEig,r,dPg uyecercsFtBi;eVNt\ni  u\nKuEew \ngCUup s tf wwo \nern ndaBy& nNbnLkmb \n W kiekmafdm\ndI, mtl.nn,Tlr wntAYo uaII...`

### Transformer
- Temperature 0.5
  - Sample 1: `To be or not to be s y oe he the ge s ase f theee me ate ste n t th ye te ee e th thele yteee the ge h me nth f ul t th hn s th t gePthe he ge s aage the tashe th nothed te thhe `
  - Sample 2: `To be or not to be s t t they the the utheyour ty he a s te ie gar ore th he thour athe tho th we yteiW whanen th ie he th atheithe me n th ath s t thre s n ug the t\nthga thy te `
  - Sample 3: `To be or not to be the the s te ghesuf th the d her ta t athe, th the th s the thethi the aer t\n te ath th who fere thath  tN thy b h h the me dthe t mthe the the tA w uth nn the`
- Temperature 1.0
  - Sample 1: `To be or not to be wiy:oee,yer f guN\ntapemfr neeehme aler tsonOY ts ye af eene.-d thelimygenep\n Kegyeh mihno, f,ulme vd mnlys ind giPthw X OR as aagcanitheS;he y-\nnotbsd ge'\nth...`
  - Sample 2: `To be or not to bele t t vrey oFce d l thpyouWteride ad ho pi, gar orediiIeyiUggcrr,QuiOe t th we ytQIW whan.\n  t id bebost\ntroith: meSn th a\no sCbtaarusB'nbugi, ghu\n$yg: zty,V...`
  - Sample 3: `To be or not to bee\nne\nt,kyseco ghrsufbyh Tpe$Rlhzruta EEegar,y gyuye sh sFteieeV t\ni tuten rw \nghe ans thowwo ferf ndayy& nN ntym l\n W uiek\nafdtedI, mtl.nn,Ter wytAYo uaIInn...`
- Temperature 1.5
  - Sample 1: `To be or not to be wiy:oeo,yer fnguN\n-apZmfrtneeehme aler tsonOY ts ye af eOne.-d hzllimygehep\n KegyJy mphnE, f,ulme vdkhn'KsNindo iPthw X OR;asgaagcN.icFfS;he y-\nnotbsdje '\nrh...`
  - Sample 2: `To be or not to belBUe IUvrfy HFce dul thpyouWceridvyad hovpi,yIarCorekiiIJyiUggcrr,QucOe t tb wehytQIW whan.\nc o&id3bebostatroid\n: meSn tu f\nimsCbtaarusB'nbugi, Jru\n$yg: zty,V...`
  - Sample 3: `To be or not to beeTne\n,,kyseqomeAr.ufbyh Tpe$RlhzruI: EEeg,r,yPgyuye srkyFtBi;eV$t\ni tutKuEew \ngCUup s tQowwo ferf ndaBy& nNOnLymb h W kiekmafdt\nPI, mtl.nn,Ter wytAYo uaIInn.E...`

## Short Analysis

At lower temperature, both models stay closer to the prompt and produce more repetitive but structured text. As temperature increases, the samples become more diverse and less grammatical. Across the current checkpoints, the Transformer is the stronger quantitative model and the cleaner sampler at lower temperatures, while the LSTM becomes noisier more quickly at higher temperatures.
