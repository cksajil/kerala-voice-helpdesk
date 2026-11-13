# Automatically built KB entries (`data/kb_auto.jsonl`)

Built by `python -m ingest.kb_auto` from `data/canon.jsonl` (cached Gemini output over OCR of the citizen charters). Curated entries (`data/kb.jsonl`, 50) are unchanged.

- auto entries: **137**, verify=true: **63** (46%)
- from official government domains only: 137
- duplicate_of_auto 6, duplicate_of_curated 28, kept 137, one_charter 244, possible_duplicates 24

| service | entries | verify=true |
|---|---|---|
| lsgd_other | 38 | 17 |
| health | 27 | 7 |
| certificate | 21 | 11 |
| trade_licence | 14 | 10 |
| general | 9 | 3 |
| pet | 7 | 3 |
| building_permit | 5 | 3 |
| birth_death | 5 | 4 |
| property_tax | 4 | 1 |
| marriage | 3 | 3 |
| pension | 2 | 1 |
| unemployment | 1 | 0 |
| grievance | 1 | 0 |

## Entries

| id | service | charters | verify | title |
|---|---|---|---|---|
| auto-g5776c5d28a | marriage | 8 | yes | സാധുക്കളായ വിധവകളുടെ പെൺമക്കളുടെ വിവാഹത്തിനുള്ള ധനസഹായം |
| auto-ga6581711a3 | building_permit | 8 | yes | കെട്ടിട നിർമ്മാണ പെർമിറ്റും പുതുക്കലും |
| auto-gf3c2e97e65 | lsgd_other | 8 | yes | ഇൻഷുറൻസ് സേവനങ്ങൾ |
| auto-g4aaee92266 | birth_death | 7 | yes | വിദേശത്തു നടന്ന ജനനങ്ങളുടെ രജിസ്ട്രേഷൻ |
| auto-g9073ebee18 | unemployment | 7 | no | തൊഴിൽ കാർഡ് |
| auto-ge08d6d43bf | certificate | 7 | yes | വയസ്സ് തെളിയിക്കുന്ന സർട്ടിഫിക്കറ്റ് |
| auto-g094a422ab9 | pension | 6 | no | അഗതി പെൻഷൻ (വിധവ / വിവാഹമോചിതർ) |
| auto-g3d0a18ef31 | property_tax | 6 | yes | ഒഴിഞ്ഞു കിടക്കുന്ന കെട്ടിടങ്ങൾക്ക് നികുതി ഇളവ് ചെയ്യൽ |
| auto-g7031eac968 | property_tax | 6 | no | കെട്ടിട ഉടമസ്ഥാവകാശ സർട്ടിഫിക്കറ്റ് / കൈമാറ്റം |
| auto-g625f9aed68 | grievance | 5 | no | പൊതു പരാതികൾ |
| auto-gcf55a260ad | certificate | 5 | yes | വാസയോഗ്യമായ വീടില്ലാത്തതിന്റെ സർട്ടിഫിക്കറ്റ് |
| auto-gf42bbd96ff | lsgd_other | 5 | yes | ട്യൂട്ടോറിയൽ പാരാമെഡിക്കൽ സ്ഥാപനങ്ങളുടെ രജിസ്ട്രേഷൻ |
| auto-g09016a4caa | trade_licence | 4 | no | ഫാക്ടറികൾ ആരംഭിക്കുന്നതിനുള്ള അനുമതി |
| auto-g224c695cf2 | pet | 4 | yes | പക്ഷി മൃഗാദികളുടെ ചികിത്സ |
| auto-g23c351ce62 | trade_licence | 4 | no | വ്യവസായ വ്യാപാര സ്ഥാപനങ്ങളുടെ ലൈസൻസ് |
| auto-g2b066a30fd | lsgd_other | 4 | no | റോഡ് മുറിച്ച് പൈപ്പ് ഇടുന്നതിനുള്ള അനുമതി |
| auto-g34bb9c356a | trade_licence | 4 | yes | ലൈവ് സ്റ്റോക്ക് ഫാം ലൈസൻസ് |
| auto-g37328a8eb9 | lsgd_other | 4 | no | കൃത്രിമ ബീജസങ്കലനം |
| auto-g4be2c294ca | certificate | 4 | no | ഗസറ്റഡ് ഓഫീസർ സാക്ഷ്യപ്പെടുത്തൽ |
| auto-g94b5c2636f | health | 4 | no | കുടുംബാസൂത്രണ പ്രവർത്തനങ്ങൾ |
| auto-g9e537aec8d | certificate | 4 | yes | തിരിച്ചറിയൽ സർട്ടിഫിക്കറ്റ് |
| auto-gad17761b98 | health | 4 | yes | പ്രതിരോധ കുത്തിവയ്പ്പ് |
| auto-gb97c5d3e35 | certificate | 4 | yes | ലൈഫ് സർട്ടിഫിക്കറ്റ് |
| auto-gce147fc7ce | lsgd_other | 4 | yes | കാർഷിക വിളകളുടെ കൃഷിരീതികളെക്കുറിച്ചുള്ള വിവരവും ഉപദേശവും |
| auto-ge035127c6a | lsgd_other | 4 | yes | പദ്ധതി ആനുകൂല്യങ്ങൾ |
| auto-gf9cf748bf8 | health | 4 | yes | സ്വകാര്യ ആശുപത്രികളും പാരാമെഡിക്കൽ സ്ഥാപനങ്ങളും രജിസ്ട്രേഷൻ |
| auto-g0b38f2a248 | lsgd_other | 3 | no | വിവിധ വിളകൾക്കായുള്ള സർക്കാർ ആനുകൂല്യങ്ങൾ |
| auto-g1fa77d5416 | health | 3 | yes | സാന്ത്വന പരിചരണവും ഭവന സന്ദർശനവും |
| auto-g219886c2f5 | property_tax | 3 | no | കെട്ടിട നികുതി രജിസ്റ്ററിൽ താമസക്കാരന്റെ പേര് ചേർക്കൽ |
| auto-g26b602d08d | health | 3 | yes | ഒ.പി. ചികിത്സയും ഔഷധവും |
| auto-g38d6c5ce16 | certificate | 3 | no | വിടുതൽ സർട്ടിഫിക്കറ്റ് |
| auto-g48ed1f4887 | certificate | 3 | no | കുടുംബബന്ധം സർട്ടിഫിക്കറ്റ് |
| auto-g5f4a00ceb3 | lsgd_other | 3 | yes | ലൈവ് സ്റ്റോക്ക് ഫാം അനുമതി |
| auto-g677bb570bc | health | 3 | no | അങ്കണവാടി അനുപൂരക പോഷകാഹാര പരിപാടി |
| auto-g68d91d3b9b | lsgd_other | 3 | no | വിള ഇൻഷുറൻസ് |
| auto-g805531fa9a | health | 3 | no | മാതൃശിശു സംരക്ഷണം |
| auto-g9660cb9b63 | lsgd_other | 3 | no | കാർഷിക പദ്ധതി ആനുകൂല്യങ്ങൾ |
| auto-g9d8bdb7704 | certificate | 3 | no | കെട്ടിടത്തിന്റെ ഏജ് സർട്ടിഫിക്കറ്റ് |
| auto-ga6e3fb689f | health | 3 | no | ആരോഗ്യ സേവനങ്ങളും പരിശോധനകളും |
| auto-gaf96725235 | property_tax | 3 | no | വിമുക്തഭടന്മാർക്കുള്ള കെട്ടിട നികുതി ഒഴിവാക്കൽ |
| auto-gb27ac91abd | health | 3 | no | കിടത്തി ചികിത്സാ വിഭാഗം |
| auto-gb327cbddae | pet | 3 | no | മൃഗങ്ങൾക്കുള്ള ലൈസൻസ് / പ്രതിരോധം |
| auto-gd4070f63e5 | health | 3 | no | ജനറൽ ഒ.പി / വൃദ്ധരോഗീ പരിചരണം |
| auto-gd96ff857ec | certificate | 3 | no | പുനർ വിവാഹം കഴിച്ചിട്ടില്ലാത്ത സർട്ടിഫിക്കറ്റ് |
| auto-gdfa0b82ce0 | certificate | 3 | yes | ബി.പി.എൽ. സർട്ടിഫിക്കറ്റ് |
| auto-gdfaf55ae4e | certificate | 3 | yes | ചികിത്സാ സംബന്ധമായ സർട്ടിഫിക്കറ്റ് |
| auto-ge785332755 | marriage | 3 | yes | വിവാഹ ധനസഹായം |
| auto-gec17d46e08 | health | 3 | no | ആരോഗ്യ വകുപ്പ് സേവനങ്ങളും മരുന്ന് വിതരണവും |
| auto-gf0990a029c | lsgd_other | 3 | no | ജനിവിജയ ബോധവത്കരണ ക്ലാസ്സുകളും ക്യാമ്പുകളും |
| auto-gf643224fc4 | lsgd_other | 3 | yes | പരസ്യങ്ങൾ പ്രദർശിപ്പിക്കുന്നതിനുള്ള അനുമതി |
| auto-gfb66e3bd42 | pet | 3 | no | വളർത്തു മൃഗങ്ങളുടെ ഇൻഷുറൻസ് സംരക്ഷണം |
| auto-gfdc8493405 | pet | 3 | yes | മൃഗങ്ങളുടെ രോഗനിർണ്ണയവും ചികിത്സയും |
| auto-g01338592ba | lsgd_other | 2 | no | കീടനാശിനി വളം ഗുണനിലവാര പരിശോധന |
| auto-g0880674b34 | certificate | 2 | yes | ജലസാമ്പിൾ പരിശോധനാ സർട്ടിഫിക്കറ്റ് |
| auto-g0cb22c6bdf | lsgd_other | 2 | no | നെൽകൃഷി ഉല്പാദന ബോണസ് |
| auto-g0cb38d9c29 | pension | 2 | yes | ഭവന നിർമ്മാണം (പി.എം.എ.വൈ / ലൈഫ്) |
| auto-g12c6b52a55 | lsgd_other | 2 | yes | പൊതുമരാമത്ത് പ്രവൃത്തികളുടെ എസ്റ്റിമേറ്റും ബില്ലും |
| auto-g1317235ae3 | health | 2 | yes | ക്ലിനിക്കൽ സൈക്കോളജിസ്റ്റ് സേവനം |
| auto-g145d6cf5eb | general | 2 | no | ലിങ്കേജ് വായ്പ പദ്ധതി |
| auto-g157cd1804a | lsgd_other | 2 | no | ബോധവത്ക്കരണം |
| auto-g1645082140 | birth_death | 2 | yes | ജനന-മരണ രജിസ്ട്രേഷൻ നോൺ അവൈലബിലിറ്റി സർട്ടിഫിക്കറ്റ് |
| auto-g17cfd3ae6e | general | 2 | no | ജെ.എൽ.ജി (ജോയിന്റ് ലയബിലിറ്റി ഗ്രൂപ്പ്) രജിസ്ട്രേഷൻ |
| auto-g193b257202 | marriage | 2 | yes | വിവാഹ രജിസ്റ്റർ തിരുത്തൽ |
| auto-g19efdd359a | lsgd_other | 2 | yes | കുടിൽ നിർമ്മാണം |
| auto-g1a7fbd586e | trade_licence | 2 | yes | സ്വകാര്യ നഴ്സറി ലൈസൻസ് |
| auto-g1b465cc34c | general | 2 | yes | വീട് മെയിന്റനൻസ് കിണർ മൂല്യനിർണ്ണയം |
| auto-g1fe637ab9e | health | 2 | no | അംഗൻവാടി സേവനങ്ങളും ആരോഗ്യ പരിശോധനയും |
| auto-g22eb708d1d | pet | 2 | no | ഓമനമൃഗങ്ങളുടെ ചികിത്സാ ഫീസ് |
| auto-g24235f7da7 | trade_licence | 2 | yes | രാസവള കീടനാശിനി വിൽപ്പന ലൈസൻസ് |
| auto-g25af61834a | health | 2 | no | രോഗപ്രതിരോധ കുത്തിവെയ്പ്പും പ്രഥമ ശുശ്രൂഷയും |
| auto-g2686816d8d | certificate | 2 | no | പിന്തുടർച്ചാവകാശ സർട്ടിഫിക്കറ്റ് |
| auto-g26f7fb2fbc | lsgd_other | 2 | no | അങ്കണവാടി വിദ്യാഭ്യാസം |
| auto-g2b2c9f878e | trade_licence | 2 | yes | സ്വകാര്യ വാഹന സ്റ്റാൻഡുകൾക്കുള്ള ലൈസൻസ് |
| auto-g2c3554ee34 | health | 2 | no | ജനറൽ ഒ.പി |
| auto-g2e2054ca56 | lsgd_other | 2 | no | വിത്തുകളുടെയും നടീൽ വസ്തുക്കളുടെയും വിതരണം |
| auto-g303e17db05 | lsgd_other | 2 | no | അസിസ്റ്റന്റ് എഞ്ചിനീയർ നിർവ്വഹണം നടത്തുന്ന പ്രോജക്ട് ആനുകൂല്യങ്ങൾ |
| auto-g31e45c4a2d | general | 2 | yes | ഭവന പുനരുദ്ധാരണം |
| auto-g327c6a02a7 | lsgd_other | 2 | no | ലൈവ് സ്റ്റോക്ക് ഇൻസ്പെക്ടറുടെ സേവനം |
| auto-g3b4ad0b0e0 | lsgd_other | 2 | no | അത്യുല്പാദന ശേഷിയുള്ള വിത്തുകളും നാടൻ വസ്തുക്കളും വിതരണം |
| auto-g3e1dfceacd | health | 2 | no | ദേശീയ മാതൃസഹായ പദ്ധതി |
| auto-g3e42870005 | lsgd_other | 2 | yes | അടിയന്തിര മണ്ണ് പരിശോധന ഫലം ലഭിക്കാൻ |
| auto-g3fc9d402fd | health | 2 | no | പകർച്ചവ്യാധി നിയന്ത്രണവും ബോധവത്കരണവും |
| auto-g43b3248ba8 | trade_licence | 2 | yes | കീടനാശിനി ജൈവവളം ജില്ലാതല വിതരണ ലൈസൻസ് |
| auto-g4a9fc66a85 | health | 2 | yes | കുട്ടികളുടെ കുത്തിവയ്പ്പ് |
| auto-g520cd8ce7f | lsgd_other | 2 | no | കാർഷിക വിളകളുടെ കീടരോഗ പരിശോധന |
| auto-g5253758106 | certificate | 2 | no | പ്രവേശന രജിസ്റ്റർ കോപ്പി |
| auto-g5b48b3bf86 | health | 2 | no | ചികിത്സയും നിർദ്ദേശങ്ങളും |
| auto-g5ed5360620 | building_permit | 2 | yes | കിണർ നിർമ്മാണ പെർമിറ്റ് |
| auto-g5f48773e93 | certificate | 2 | yes | ആർ.സി. നഷ്ടപ്പെട്ടവർക്കുള്ള സർട്ടിഫിക്കറ്റ് |
| auto-g6072afc8b4 | health | 2 | no | നേത്രപരിശോധന |
| auto-g62826c38ac | general | 2 | no | രേഖകൾ സാക്ഷ്യപ്പെടുത്തൽ |
| auto-g68d413a053 | health | 2 | no | രോഗപ്രതിരോധ കുത്തിവയ്പ് |
| auto-g6b99eb5e79 | lsgd_other | 2 | no | വന്ധ്യതാ നിവാരണ ക്യാമ്പുകൾ |
| auto-g6d58527516 | lsgd_other | 2 | yes | നീർത്തിട വികസന പ്രവൃത്തികൾ |
| auto-g6e8c2b7c8c | lsgd_other | 2 | no | നെൽകൃഷിക്കുള്ള ഉൽപാദക ബോണസ് |
| auto-g706060a53d | health | 2 | no | കിടപ്പുരോഗികൾക്കുള്ള ഗൃഹപരിചരണവും സാന്ത്വന ചികിത്സയും |
| auto-g770c948606 | certificate | 2 | yes | സർട്ടിഫിക്കറ്റ് പകർപ്പ് സാക്ഷ്യപ്പെടുത്തൽ |
| auto-g7cff56e295 | building_permit | 2 | no | കെട്ടിട നമ്പർ നൽകലും ഒക്കുപെൻസി സർട്ടിഫിക്കറ്റും |
| auto-g7eb862cf1d | certificate | 2 | no | അഡ്മിഷൻ രജിസ്റ്റർ പകർപ്പ് നൽകൽ |
| auto-g80b56852b8 | trade_licence | 2 | yes | പി.പി.ആർ ആക്ട് പ്രകാരമുള്ള ലൈസൻസ് |
| auto-g840e5e4785 | birth_death | 2 | yes | ജനന മരണ രജിസ്ട്രേഷനിലെ തിരുത്തലുകൾ |
| auto-g85630f40e0 | health | 2 | yes | കന്നുകാലികളുടെ കൃത്രിമ ബീജസങ്കലനം |
| auto-g86f99a0b3f | lsgd_other | 2 | yes | കോഴിവസന്ത പ്രതിരോധ കുത്തിവയ്പ് |
| auto-g8cbd11c941 | lsgd_other | 2 | no | കൃഷിയുമായി ബന്ധപ്പെട്ട മറ്റ് ആനുകൂല്യങ്ങൾ |
| auto-g8ddacc11b9 | trade_licence | 2 | yes | മരാധിഷ്ഠിത വ്യവസായങ്ങൾക്കുള്ള ലൈസൻസ് |
| auto-g8ece90666f | certificate | 2 | no | പേരിലോ വീട്ടുപേരിലോ മാറ്റം വന്നതിനുള്ള തിരിച്ചറിയൽ സർട്ടിഫിക്കറ്റ് |
| auto-g8f64bc36ca | general | 2 | no | മൈക്രോഫിനാൻസ് ഗ്രൂപ്പ് രൂപീകരണം |
| auto-g92481de82a | lsgd_other | 2 | no | പ്രകൃതിക്ഷോഭ ദുരിതാശ്വാസം |
| auto-g98971f9957 | birth_death | 2 | no | ജനനത്തീയതി സർട്ടിഫിക്കറ്റ് (വോട്ടർ പട്ടിക/റേഷൻകാർഡ്) |
| auto-g99160ed47b | lsgd_other | 2 | no | കാർഷിക ആവശ്യത്തിനുള്ള വൈദ്യുതി കണക്ഷൻ |
| auto-g9993298f6d | health | 2 | no | പോഷകാഹാര വിതരണം |
| auto-g9a6cfdf881 | trade_licence | 2 | yes | ഫാക്ടറികൾക്കും വ്യവസായ സ്ഥാപനങ്ങൾക്കുമുള്ള ലൈസൻസ് |
| auto-ga041c6989f | birth_death | 2 | yes | ജനന/മരണ രജിസ്ട്രേഷന് ജില്ലാ രജിസ്ട്രാറുടെ അനുവാദം വാങ്ങൽ |
| auto-ga559146f19 | lsgd_other | 2 | yes | രേഖകൾ സാക്ഷ്യപ്പെടുത്തൽ |
| auto-ga873b74c39 | trade_licence | 2 | yes | വ്യാപാര വ്യവസായ ഡി ആന്റ് ഒ ലൈസൻസ് |
| auto-gad5e6ee6c6 | lsgd_other | 2 | yes | ഭവന നിർമ്മാണ പദ്ധതി ആനുകൂല്യ വിതരണം |
| auto-gaf05ba30b9 | lsgd_other | 2 | yes | കെട്ടിട ഉടമസ്ഥാവകാശം കൈമാറ്റം ചെയ്യൽ |
| auto-gaffd1df1b3 | building_permit | 2 | yes | ഒക്യുപൻസി സർട്ടിഫിക്കറ്റ് |
| auto-gb4932d71c3 | pet | 2 | yes | മൃഗങ്ങൾക്കും പക്ഷികൾക്കുമുള്ള ചികിത്സ |
| auto-gb7fd985508 | health | 2 | no | മാതൃ-ശിശു ആരോഗ്യ പരിപാലനം |
| auto-gb846565737 | certificate | 2 | yes | സബ്സിഡി അർഹതാ സർട്ടിഫിക്കറ്റ് |
| auto-gc54106cb94 | general | 2 | no | സ്ത്രീ പദവി സ്വയം പഠന പരിപാടി ജെണ്ടർ കോർണർ |
| auto-gc921bfa64a | trade_licence | 2 | no | രാസവളം കീടനാശിനി വില്പന ലൈസൻസ് പുതുക്കൽ |
| auto-gc9ea82e7ef | pet | 2 | no | കൃത്രിമ ബീജാധാനം |
| auto-gcd043a9729 | lsgd_other | 2 | yes | അനുപൂരക പോഷകാഹാര വിതരണം |
| auto-gcf9e102ac9 | lsgd_other | 2 | yes | കിണർ നിർമ്മാണ അനുമതി |
| auto-gd17e49d73f | trade_licence | 2 | yes | വളം നിർമ്മാണ ലൈസൻസ് |
| auto-ge1b5ab0cf5 | building_permit | 2 | no | കെട്ടിടനിർമ്മാണ അനുമതിക്കുള്ള ശുപാർശ |
| auto-ge5ae5141b0 | certificate | 2 | yes | സ്റ്റേജ് സർട്ടിഫിക്കറ്റ് |
| auto-ge680ce8169 | general | 2 | no | ഉറവിട മാലിന്യ സംസ്കരണ സംവിധാനങ്ങൾ |
| auto-geba0c50816 | health | 2 | no | ഡോക്ടറുടെ സേവനം ഒ.പി. |
| auto-gece55da913 | general | 2 | yes | ഗ്രാമപഞ്ചായത്ത് പദ്ധതികളുടെ നിർവ്വഹണം |
| auto-gf24a02809f | lsgd_other | 2 | yes | സാങ്കേതിക ഉപദേശങ്ങളും നിർദ്ദേശങ്ങളും നൽകൽ |
| auto-gf996cb7b5e | lsgd_other | 2 | no | ചെറുകിട പരിമിത വൻകിട കർഷക സർട്ടിഫിക്കറ്റ് |
| auto-gfbd015bbfc | trade_licence | 2 | no | രാസവളം ജീവാണുവളം മൊത്തവിതരണ ലൈസൻസ് |
| auto-gfc20bfa25a | health | 2 | no | രോഗ പ്രതിരോധ കുത്തിവയ്പ്പും പ്രതിരോധ നടപടിയും |
| auto-gfffa7ac037 | certificate | 2 | no | ആനുകൂല്യങ്ങൾ ലഭിച്ചിട്ടില്ലാത്തതിന്റെ സർട്ടിഫിക്കറ്റ് |

## Possible duplicates kept (check by hand)

Kept entries whose title half-matches a curated entry (0.5 <= coverage < DUP_SIM).

| id | title | closest curated | coverage |
|---|---|---|---|
| auto-g094a422ab9 | അഗതി പെൻഷൻ (വിധവ / വിവാഹമോചിതർ) | ch-pen-widow-01 | 0.70 |
| auto-g48ed1f4887 | കുടുംബബന്ധം സർട്ടിഫിക്കറ്റ് | access-01 | 0.70 |
| auto-g62826c38ac | രേഖകൾ സാക്ഷ്യപ്പെടുത്തൽ | ch-cert-pres-02 | 0.70 |
| auto-ga559146f19 | രേഖകൾ സാക്ഷ്യപ്പെടുത്തൽ | ch-cert-pres-02 | 0.70 |
| auto-ga6581711a3 | കെട്ടിട നിർമ്മാണ പെർമിറ്റും പുതുക്കലും | ch-bp-docs-01 | 0.68 |
| auto-g7031eac968 | കെട്ടിട ഉടമസ്ഥാവകാശ സർട്ടിഫിക്കറ്റ് / കൈമാറ്റം | ch-bt-own-01 | 0.67 |
| auto-g1645082140 | ജനന-മരണ രജിസ്ട്രേഷൻ നോൺ അവൈലബിലിറ്റി സർട്ടിഫിക്കറ്റ് | cr-fee-02 | 0.67 |
| auto-g840e5e4785 | ജനന മരണ രജിസ്ട്രേഷനിലെ തിരുത്തലുകൾ | cr-cert-01 | 0.67 |
| auto-gcf9e102ac9 | കിണർ നിർമ്മാണ അനുമതി | building-01 | 0.67 |
| auto-gaf96725235 | വിമുക്തഭടന്മാർക്കുള്ള കെട്ടിട നികുതി ഒഴിവാക്കൽ | ch-bt-appeal-01 | 0.62 |
| auto-g5ed5360620 | കിണർ നിർമ്മാണ പെർമിറ്റ് | ch-bp-docs-01 | 0.60 |
| auto-gaf05ba30b9 | കെട്ടിട ഉടമസ്ഥാവകാശം കൈമാറ്റം ചെയ്യൽ | ch-bt-own-01 | 0.60 |
| auto-g4aaee92266 | വിദേശത്തു നടന്ന ജനനങ്ങളുടെ രജിസ്ട്രേഷൻ | ch-bd-abroad-01 | 0.53 |
| auto-ge1b5ab0cf5 | കെട്ടിടനിർമ്മാണ അനുമതിക്കുള്ള ശുപാർശ | building-01 | 0.53 |
| auto-g3d0a18ef31 | ഒഴിഞ്ഞു കിടക്കുന്ന കെട്ടിടങ്ങൾക്ക് നികുതി ഇളവ് ചെയ്യൽ | ch-bt-appeal-01 | 0.52 |
| auto-g09016a4caa | ഫാക്ടറികൾ ആരംഭിക്കുന്നതിനുള്ള അനുമതി | building-01 | 0.50 |
| auto-g219886c2f5 | കെട്ടിട നികുതി രജിസ്റ്ററിൽ താമസക്കാരന്റെ പേര് ചേർക്കൽ | ptax-01 | 0.50 |
| auto-g9d8bdb7704 | കെട്ടിടത്തിന്റെ ഏജ് സർട്ടിഫിക്കറ്റ് | ch-bt-transfer-01 | 0.50 |
| auto-ge785332755 | വിവാഹ ധനസഹായം | cr-cert-01 | 0.50 |
| auto-g193b257202 | വിവാഹ രജിസ്റ്റർ തിരുത്തൽ | cr-cert-01 | 0.50 |
| auto-g3e1dfceacd | ദേശീയ മാതൃസഹായ പദ്ധതി | ch-pen-old-01 | 0.50 |
| auto-g770c948606 | സർട്ടിഫിക്കറ്റ് പകർപ്പ് സാക്ഷ്യപ്പെടുത്തൽ | cr-fee-02 | 0.50 |
| auto-g7eb862cf1d | അഡ്മിഷൻ രജിസ്റ്റർ പകർപ്പ് നൽകൽ | cr-fee-02 | 0.50 |
| auto-gd17e49d73f | വളം നിർമ്മാണ ലൈസൻസ് | trade-03-mfg | 0.50 |

## Not promoted

| id | title | reason |
|---|---|---|
| auto-gac684a4c64 | മരണ രജിസ്ട്രേഷനും സർട്ടിഫിക്കറ്റും | duplicate of curated cr-fee-01 |
| auto-gc61693299e | ജനന മരണ സർട്ടിഫിക്കറ്റ് | duplicate of curated cr-cert-01 |
| auto-g7a61f6077c | ജനന രജിസ്റ്ററിൽ കുട്ടിയുടെ പേര് ചേർക്കൽ | duplicate of curated cr-fee-02 |
| auto-g178eddb240 | ദത്തെടുക്കുന്ന കുട്ടികളുടെ ജനന രജിസ്ട്രേഷൻ | duplicate of curated ch-bd-adopt-01 |
| auto-g13cc2aa21f | ജനന/മരണം വൈകി രജിസ്റ്റർ ചെയ്യൽ | duplicate of curated cr-fee-01 |
| auto-g3959f4406c | ജനന മരണ വൈകിയുള്ള രജിസ്ട്രേഷൻ | duplicate of curated cr-fee-01 |
| auto-gfec7624200 | ജനനം/മരണം രജിസ്റ്റർ ചെയ്തിട്ടില്ലെന്ന സർട്ടിഫിക്കറ്റ് | duplicate of curated ch-bd-late-01 |
| auto-gc806d98af4 | ജനന രജിസ്റ്ററിൽ പേര് ചേർക്കൽ | duplicate of curated cr-fee-02 |
| auto-ge68894ca97 | കെട്ടിട നിർമ്മാണ പെർമിറ്റ് സേവനങ്ങൾ | duplicate of curated ch-bp-docs-01 |
| auto-g486a79f175 | കെട്ടിട നിർമ്മാണ അനുമതി | duplicate of curated building-01 |
| auto-ga5d95c177b | താമസ സർട്ടിഫിക്കറ്റ് | duplicate of curated ch-bt-res-01 |
| auto-gdee42aaa0c | സ്വഭാവ സർട്ടിഫിക്കറ്റ് | duplicate of curated ch-cert-pres-02 |
| auto-g9945266af5 | കെട്ടിട ഉടമസ്ഥാവകാശ സർട്ടിഫിക്കറ്റ് | duplicate of curated ch-bt-own-01 |
| auto-g78d23a9af2 | വിവാഹ സർട്ടിഫിക്കറ്റ് | duplicate of curated cr-cert-01 |
| auto-g97a01bf2a2 | റേഷൻ കാർഡിൽ പേരില്ലാത്ത സർട്ടിഫിക്കറ്റ് | duplicate of curated ch-cert-pres-01 |
| auto-g192dc9b7e7 | കുടുംബശ്രീ രജിസ്ട്രേഷൻ | duplicate of curated access-01 |
| auto-g76fc7be8e8 | മെഡിക്കൽ സർട്ടിഫിക്കറ്റ് | duplicate of curated ch-phc-01 |
| auto-ge44bc1169b | ഹെൽത്ത് കാർഡും ഫിറ്റ്‌നസ് സർട്ടിഫിക്കറ്റും | duplicate of curated ch-phc-01 |
| auto-gc9656b3c1d | വികലാംഗ പെൻഷൻ | duplicate of curated ch-pen-disab-01 |
| auto-g94b3c7a766 | അവിവാഹിത സർട്ടിഫിക്കറ്റ് | duplicate of curated ch-pen-unmarried-01 |
| auto-ged71a63f8f | ഇന്ദിരാഗാന്ധി ദേശീയ വാർദ്ധക്യകാല പെൻഷൻ | duplicate of curated ch-pen-old-01 |
| auto-ga6479df354 | അവിവാഹിതരായ സ്ത്രീകൾക്കുള്ള പെൻഷൻ | duplicate of curated ch-pen-unmarried-01 |
| auto-gd36ef32da6 | ഭിന്നശേഷി പെൻഷൻ | duplicate of curated ch-pen-disab-01 |
| auto-g5ddc51581e | അവിവാഹിത പെൻഷൻ | duplicate of curated ch-pen-amount-01 |
| auto-gd3a1514c7a | പട്ടി, പന്നി എന്നിവ വളർത്തുന്നതിനുള്ള ലൈസൻസ് | duplicate of curated ch-pet-01 |
| auto-g1731f9b177 | ഉടമസ്ഥാവകാശ സർട്ടിഫിക്കറ്റ് | duplicate of curated ch-bt-own-01 |
| auto-g007e6e0e5e | കെട്ടിട നികുതി ഒഴിവാക്കൽ | duplicate of curated ch-bt-appeal-01 |
| auto-gcb52ac21d3 | തൊഴിൽരഹിത വേതനം | duplicate of curated ch-unemp-01 |
| auto-g2c40b7743a | വ്യാപാര സ്ഥാപന ലൈസൻസ് | near-duplicate of auto-g23c351ce62 |
| auto-g42d138cba7 | ട്യൂട്ടോറിയൽ സ്ഥാപനങ്ങളുടെ രജിസ്ട്രേഷൻ | near-duplicate of auto-gf42bbd96ff |
| auto-g8ad81ace65 | കെട്ടിട നിർമ്മാണ പെർമിറ്റ് കാലാവധി നീട്ടൽ | near-duplicate of auto-ga6581711a3 |
| auto-g96ba374284 | തൊഴിൽ കാർഡ് രജിസ്ട്രേഷൻ | near-duplicate of auto-g9073ebee18 |
| auto-ga0e41f579f | മൃഗങ്ങൾക്കുള്ള പ്രതിരോധ കുത്തിവെയ്പ് | near-duplicate of auto-gad17761b98 |
| auto-ge221a677d8 | വിവാഹ രജിസ്ട്രേഷനിലെ തിരുത്തലുകൾ | near-duplicate of auto-g193b257202 |
