emoji_dict = {
    'odlicno': [':-)', ':)', ':-d', ':d', 'xd', '๐', '๐', '๐', '๐', '๐', '๐', '๐', '๐', '๐', '๐', '๐', '๐ฅฐ',
                '๐', '๐', '๐', 'โบ', '๐', '๐ค', '๐คฉ', '๐', '๐', '๐', '๐', '๐คค', '๐', '5+', '10+', '5.00', '5/5',
                '10/10', '๐', '๐'],
    'super': ['<3', 'โค', '๐งก', '๐', '๐', '๐', '๐', '๐ค', '๐ค', '๐ค', 'โฃ', '๐', '๐', '๐', '๐', '๐', '๐', '๐',
              '๐', '๐'],
    'lose': [':(', ':-(', '</3', '๐', '๐ค', '๐คจ', '๐', '๐', '๐ถ', '๐ฅ', '๐ค', '๐ซ', '๐ฅฑ', '๐ด', '๐', '๐', '๐',
             '๐', '๐ค', 'โน', '๐', '๐', '๐', '๐', '๐ข', '๐ญ', '๐ฆ', '๐ง', '๐จ', '๐ฉ', '๐ฌ', '๐ฐ', '๐ฅต', '๐ฑ', '๐ก',
             '๐ฅบ', '๐ฃ', '๐ ', '๐คฌ', '๐คฎ', '๐ค', '๐คข', '๐คง', '๐ฅถ', '๐ค'],
    'okej': ['๐', '๐คฃ'],
    '': [';)', ';-)', 'D:']
}

# todo: analyze more when dataset is complete
stop_words = [
    # licne zamenice
    'ja', 'mene', 'me', 'meni', 'mi', 'mnom', 'mnome', 'ti', 'tebe', 'te', 'tebi', 'tobom', 'mi', 'nas', 'nama', 'nam',
    'vi', 'vas', 'vama', 'vam', 'on', 'njega', 'ga', 'njemu', 'mu', 'njim', 'ono', 'ona', 'nje', 'je', 'njoj', 'joj',
    'nju', 'ju', 'njom', 'oni', 'njih', 'ih', 'njima', 'im', 'one',
    'sebe', 'sebi', 'sobom', 'se',
    # upitne zamenice
    'ko', 'sta', 'koga', 'cega', 'kome', 'cemu',
    # neodredjene zamenice
    'neko', 'nesto', 'nekog', 'nekoga', 'neceg', 'necega', 'nekim', 'nekime', 'necim', 'necime', 'nekome', 'necemu',
    'nekom', 'necem',
    # prisvojne pridevske zamenice
    'moj', 'moja', 'moje', 'tvoj', 'tvoja', 'tvoje', 'njegov', 'njegova', 'njegovo', 'njen', 'njena', 'njeno', 'nas',
    'nasa', 'nase', 'vas', 'vasa', 'vase', 'njihov', 'njihova', 'njihovo', 'svoj', 'svoja', 'svoje',
    # pokazne pridevske zamenice
    'ovaj', 'ova', 'ovo', 'onaj', 'taj', 'ta', 'to', 'ovakav', 'ovakva', 'ovakvo', 'ovoliki', 'ovolika', 'ovoliko',
    'onoliki', 'onolika', 'onoliko', 'toliki', 'tolika', 'toliko',
    # upitno-odnosne prisvojne zamenice
    'koji', 'koja', 'koje', 'ciji', 'cije', 'kakav', 'kakva', 'kakvo', 'koliki', 'kolika', 'koliko',
    # predlozi
    'pred', 'nad', 'u', 'iz', 'od', 'do', 'zbog', 'radi', 'sa', 'o', 's', 'na', 'uz', 'ka', 'pod', 'pri', 'za', 'kod',
    'oko', 'pored', 'iznad',
    # veznici - izbaceno 'iako', 'ni', 'niti', 'sve', 'a', 'ali'
    'i', 'pa', 'te', 'no', 'nego', 'vec', 'jer', 'ako', 'osim', 'sem', 'kao', 'da', 'mada', 'premda', 'dok',
    'sav', 'sva', 'svo',
    # recce - izbaceno 'cak', 'tek', 'pritom'
    'dakle', 'li', 'zar', 'inace', 'takodje', 'takode', 'uostalom', 'neka', 'nek',
    # prilozi - izbaceno 'skroz', 'koliko', 'zato', 'odozgo'
    'kako', 'kada', 'onda', 'sad', 'negde', 'odavde', 'inace',
    # skracenice
    'din', 'god', 'cm', 'kg', 'kilo', 'kilogram', 'min', 'gr', 'ps',
    # netacne formulacije
    'ustvari', 'odole', 'kolko', 'al',
    'com', 'http', 'https'
]

# used for stemming
# source - https://github.com/nikolamilosevic86/SerbianStemmer
suffixes = {
    'ovnicki': '',
    'ovnicka': '',
    'ovnika': '',
    'ovniku': '',
    'ovnicke': '',
    'kujemo': '',
    'ovacu': '',
    'ivacu': '',
    'isacu': '',
    'dosmo': '',
    'ujemo': '',
    'ijemo': '',
    'ovski': '',
    'ajuci': '',
    'icizma': '',
    'ovima': '',
    'ovnik': '',
    'ognu': '',
    'inju': '',
    'enju': '',
    'cicu': '',
    'stva': '',
    'ivao': '',
    'ivala': '',
    'ivalo': '',
    'skog': '',
    'ucit': '',
    'ujes': '',
    'uces': '',
    'oces': '',
    'osmo': '',
    'ovao': '',
    'ovala': '',
    'ovali': '',
    'ismo': '',
    'ujem': '',
    'esmo': '',
    'asmo': '',
    'zemo': '',
    'cemo': '',
    'bemo': '',
    'ovan': '',
    'ivan': '',
    'isan': '',
    'uvsi': '',
    'ivsi': '',
    'evsi': '',
    'avsi': '',
    'suci': '',
    'uste': '',
    'ace': 'ak',
    'uze': 'ug',
    'aze': 'ag',
    'aci': 'ak',
    'oste': '',
    'aca': '',
    'enu': '',
    'enom': '',
    'enima': '',
    'eta': '',
    'etu': '',
    'etom': '',
    'adi': '',
    'nju': 'nj',
    'lju': '',
    'lja': '',
    'lji': '',
    'lje': '',
    'ljom': '',
    'ljama': '',
    'zi': 'g',
    'etima': '',
    'ac': '',
    'beci': 'beg',
    'nem': '',
    'nes': '',
    'ne': '',
    'nemo': '',
    'nimo': '',
    'nite': '',
    'nete': '',
    'nu': '',
    'ce': '',
    'ci': '',
    'cu': '',
    'ca': '',
    'cem': '',
    'cima': '',
    'scu': 's',
    'ara': 'r',
    'iste': '',
    'este': '',
    'aste': '',
    'ujte': '',
    'jete': '',
    'jemo': '',
    'jem': '',
    'jes': '',
    'ijte': '',
    'inje': '',
    'anje': '',
    'acki': '',
    'inja': '',
    'alja': '',
    'nog': '',
    'omu': '',
    'emu': '',
    'uju': '',
    'iju': '',
    'sko': '',
    'eju': '',
    'ahu': '',
    'ucu': '',
    'icu': '',
    'ecu': '',
    'acu': '',
    'ocu': '',
    'izi': 'ig',
    'ici': 'ik',
    'tko': 'd',
    'tka': 'd',
    'ast': '',
    'tit': '',
    'nus': '',
    'ces': '',
    'cno': '',
    'cni': '',
    'cna': '',
    'uto': '',
    'oro': '',
    'eno': '',
    'ano': '',
    'umo': '',
    'smo': '',
    'imo': '',
    'emo': '',
    'ulo': '',
    'slo': '',
    'ila': '',
    'ilo': '',
    'ski': '',
    'ska': '',
    'elo': '',
    'njo': '',
    'ovi': '',
    'evi': '',
    'uti': '',
    'iti': '',
    'eti': '',
    'ati': '',
    'vsi': '',
    'ili': '',
    'eli': '',
    'ali': '',
    'uji': '',
    'nji': '',
    'uci': '',
    'sci': '',
    'eci': '',
    'oci': '',
    'ove': '',
    'eve': '',
    'ute': '',
    'ste': '',
    'nte': '',
    'kte': '',
    'jte': '',
    'ite': '',
    'ete': '',
    'use': '',
    'ese': '',
    'ase': '',
    'une': '',
    'ene': '',
    'ule': '',
    'ile': '',
    'ele': '',
    'ale': '',
    'uke': '',
    'tke': '',
    'ske': '',
    'uje': '',
    'tje': '',
    'sce': '',
    'ice': '',
    'ece': '',
    'uce': '',
    'oce': '',
    'ova': '',
    'eva': '',
    'ava': 'av',
    'uta': '',
    'ata': '',
    'ena': '',
    'ima': '',
    'ama': '',
    'ela': '',
    'ala': '',
    'aka': '',
    'aja': '',
    'jmo': '',
    'oga': '',
    'ega': '',
    'aฤa': '',
    'oca': '',
    'aba': '',
    'cki': '',
    'ju': '',
    'hu': '',
    'ut': '',
    'it': '',
    'et': '',
    'at': '',
    'us': '',
    'is': '',
    'es': '',
    'uo': '',
    'no': '',
    'mo': '',
    'lo': '',
    'io': '',
    'eo': '',
    'ao': '',
    'un': '',
    'an': '',
    'om': '',
    'ni': '',
    'im': '',
    'em': '',
    'uk': '',
    'uj': '',
    'oj': '',
    'li': '',
    'uh': '',
    'oh': '',
    'ih': '',
    'eh': '',
    'ah': '',
    'og': '',
    'eg': '',
    'te': '',
    'se': '',
    'le': '',
    'ke': '',
    'ko': '',
    'ka': '',
    'ti': '',
    'he': '',
    'ad': '',
    'ec': '',
    'na': '',
    'ma': '',
    'ul': '',
    'ku': '',
    'la': '',
    'nj': 'nj',
    'lj': 'lj',
    'ha': '',
    'a': '',
    'e': '',
    'u': '',
    's': '',
    'o': '',
    'i': '',
    'j': '',
}

# source - https://github.com/nikolamilosevic86/SerbianStemmer (changed logic slightly)
verbs = [
    'bih',
    'bi',
    'bismo',
    'biste',
    'bise',
    'budem',
    'budes',
    'bude',
    'budemo',
    'budete',
    'budu',
    'bio',
    'bila',
    'bili',
    'bile',
    'biti',
    'bijah',
    'bijase',
    'bijasmo',
    'bijaste',
    'bijahu',
    'bese',

    'sam',
    'si',
    'je',
    'smo',
    'ste',
    'su',
    'jesam',
    'jesi',
    'jeste',
    'jesmo',
    'jesu',

    'cu',
    'ces',
    'ce',
    'cemo',
    'cete',
    'hocu',
    'hoces',
    'hocemo',
    'hocete',
    'hoce',
    'hteo',
    'htela',
    'hteli',
    'htelo',
    'htele',
    'htedoh',
    'htede',
    'htedosmo',
    'htedoste',
    'htedose',
    'hteh',
    'hteti',
    'htejuci',
    'htevsi',

    'mogu',
    'mozes',
    'moze',
    'mozemo',
    'mozete',
    'mogao',
    'mogli',
    'moci'

    'imam',
    'imas',
    'ima',
    'imamo',
    'imate',
    'imaju',
    'imali',
    'imao',
    'imale',
    'imalo',
    'imace'
]


def replace_special_chars(review):
    for c in '.,!?-:();*="\'\\/#':
        review = review.replace(c, ' ')
    return review


def replace_emoji(review):
    for sentiment, emoji_list in emoji_dict.items():
        for emoji in emoji_list:
            review = review.replace(emoji, ' ' + sentiment + ' ')
    return review


def lem_stem(word):
    if word in verbs:
        return ''
    for key in suffixes:
        if word.endswith(key) and len(word) - len(key) > 2:
            return word[:-len(key)] + suffixes[key]
    return word


def remove_whitespace(text):
    return " ".join(text.split())


def preprocess(review):
    in_letters = 'ฤฤฤลพลก'
    out_letters = 'ccdzs'
    trans_tab = str.maketrans(in_letters, out_letters)
    review = str.lower(review)
    review = replace_emoji(review)
    review = replace_special_chars(review)
    all_words = review.split(' ')
    update_list = []
    for word in all_words:
        if word != '' and word not in stop_words:
            processed_word = word.translate(trans_tab)
            lemmed_stemmed = lem_stem(processed_word)
            if lemmed_stemmed != '':
                update_list.append(processed_word)
    return update_list


def df_preprocess(data):
    reviews = data['review']
    update_review = []
    for review in reviews:
        update_list = preprocess(review)
        update_review.append(' '.join(update_list))
    data['review'] = update_review
    return data
