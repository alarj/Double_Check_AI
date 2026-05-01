# Post-check reeglimudel

## Eesmärk
Dokument kirjeldab post-check otsustusreegleid ühtse tehnilise spetsifikatsioonina, et API, UI ja testid kasutaksid sama loogikat.

Post-check on kaheosaline:
- 4a sisuline kontroll (`/post-check-quality`)
- 4b turvakontroll (`/post-check-security`)

Koondkontroll (`/post-check`) ühendab mõlema tulemuse.

## Mõisted
- **original_user_input**: kasutaja algne küsimus.
- **normalized_query**: normaliseeritud küsimus.
- **context**: retrievali järel saadud kontekst.
- **ai_response**: põhimudeli vastus.
- **sources_returned_raw**: retrievali kandidaatide tehniline info (metadata + `selected` + filtrid).
- **ALLOWED**: kontroll läbitud.
- **BLOCKED**: kontroll ebaõnnestus.

## Sisendid

### 4a sisuline kontroll
Kasutab välju:
- `original_user_input`
- `normalized_query`
- `context`
- `ai_response`
- `quality_model` (või fallback `model`)
- `threads`, `timeout`

### 4b turvakontroll
Kasutab välju:
- `original_user_input`
- `normalized_query`
- `context`
- `ai_response`
- õigused: `secret`, `allow_all_subjects`, `allow_personal_data`, `allowed_subject_ids`, `allowed_tenant_ids`
- `sources_returned_raw`
- `security_model` (või fallback `model`)
- `threads`, `timeout`

## Otsustusloogika

## 4a sisuline kontroll
4a kontrollib faktitäpsust ja vastavust kontekstile.

### 4a hard-rule reeglid (deterministlikud)
Hard-rule’d rakenduvad enne LLM-kontrolli:

1. **Tühi kontekst + sisuline vastus**
   - tingimus: `context` on tühi
   - ja `ai_response` ei ole täpselt `Esitatud kontekstis info puudub.`
   - tulemus: `BLOCKED`

2. **Kontekstivälised arvulised faktid**
   - tingimus: `ai_response` sisaldab arvulisi väärtusi, mida `context` ei sisalda
   - tulemus: `BLOCKED`

Kui hard-rule ei rakendu, tehakse LLM-põhine sisuline kontroll `POST_CHECK_PROMPT` alusel.

### 4a LLM-kontrolli eesmärk
- iga faktiline väide peab olema kontekstiga kaetud
- kontekstis nähtamatuid konkreetseid fakte ei tohi lisada
- väljamõeldud või ületäpseid viiteid ei tohi luua
- vastus peab vastama küsimusele

## 4b turvakontroll
4b kontrollib õigusi ja andmekaitset.

### 4b hard-rule reeglid (deterministlikud)
Hard-rule’d rakenduvad enne LLM-kontrolli:

1. **Salajase info kasutus ilma õiguseta**
   - tingimus: `secret=false`
   - ja `sources_returned_raw` sisaldab `selected=true` kandidaati, mille `classification_level=secret`
   - ja vastus ei ole `Esitatud kontekstis info puudub.`
   - tulemus: `BLOCKED`

2. **Maskeerimata isikuandmed**
   - tingimus: `allow_personal_data=false`
   - ja vastus sisaldab maskeerimata isikutuvastust (`subject_id`/`counterparty_id`/`personal_id` metadata põhjal või tekstimuster)
   - tulemus: `BLOCKED`

3. **Subject-piirangu rikkumine**
   - tingimus: `allow_all_subjects=false`
   - ja `allowed_subject_ids` on antud
   - ja `selected=true` kandidaadi `subject_id` ei kuulu lubatud hulka
   - tulemus: `BLOCKED`

4. **Tenant-piirangu rikkumine**
   - tingimus: `allowed_tenant_ids` on antud
   - ja `selected=true` kandidaadi `tenant_id` ei kuulu lubatud hulka
   - tulemus: `BLOCKED`

Kui hard-rule ei rakendu, tehakse LLM-põhine turvakontroll `POST_CHECK_SECURITY_PROMPT` alusel.

### 4b LLM-kontrolli eesmärk
- tuvastada võimalikud turvalekke juhtumid, mida hard-rule ei kata
- anda lühike põhjendus `ALLOWED/BLOCKED` otsusele

## Koondotsus
Koondkontroll (`/post-check`) kasutab OR-loogikat:
- kui `quality.status == BLOCKED` või `security.status == BLOCKED` -> **lõppstaatus `BLOCKED`**
- muidu -> **lõppstaatus `ALLOWED`**

See tähendab:
- sisuline viga ei pea tähendama turvaviga
- turvaviga ei pea tähendama sisulist viga
- lõppotsus blokeerib, kui vähemalt üks kontroll ebaõnnestub

## Prioriteedid
1. Hard-rule otsus (deterministlik) on ülimuslik vastavas kontrollis.
2. Kui hard-rule ei rakendu, kasutatakse LLM-kontrolli.
3. Koondotsus arvutatakse alati 4a ja 4b tulemustest OR-loogikaga.

## Väljundi miinimumnõuded
Iga kontroll tagastab vähemalt:
- `status`: `ALLOWED` või `BLOCKED`
- `reason`: lühike põhjendus
- `model`: kasutatud mudel või `hard-rule`
- `duration`

Koondvastus sisaldab lisaks:
- `checks.quality`
- `checks.security`
- lõpp-`status`

## Seos testidega
`testing/test_post_check_use_cases.py` valideerib:
- 4a ja 4b eraldi oodatud staatused
- lokaalselt arvutatud koondstaatus
- soovi korral API koondendpointi staatus

Sellega on tagatud, et API, testid ja UI tõlgendavad sama reeglimudelit ühtmoodi.
