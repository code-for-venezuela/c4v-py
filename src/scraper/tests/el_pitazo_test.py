# TODO move tests out of this module when approved.

from scraper.tests.utils import fake_response_from_file
from scraper.scrapers.el_pitazo_scraper import ElPitazoScraper
from scraper.settings import ROOT_DIR
import os


def test_parse_ok():
    """
        Check that ElPitazoScraper parses a valid page as expected
    """
    url = "tests/html_bodies/el_pitazo_fallas_electricas_carne.html"
    test_file = os.path.join(ROOT_DIR, url)
    response = fake_response_from_file(
        test_file,
        "https://elpitazo.net/cronicas/la-fallas-electricas-han-disminuido-la-cantidad-de-carne-que-consume-el-venezolano/",
    )

    scraper = ElPitazoScraper()
    parse_output = scraper.parse(response)

    assert parse_output["body"] == get_body_for_parse_ok(), "body does not match"
    assert (
        parse_output["title"]
        == "Las fallas eléctricas han disminuido la cantidad de carne que consume el venezolano"
    ), "title does not match"
    assert parse_output["author"] == "Redacción El Pitazo", "author does not match"
    assert parse_output["tags"] == [], "tags does not match"
    assert set(parse_output["categories"]) == set(
        ["Crónicas", "Regiones"]
    ), "categorias no coinciden"


def get_body_for_parse_ok():
    return "Ganaderos de la zona del Sur del Lago de Maracaibo, área estratégica potencial para el desarrollo agropecuario de la nación, reportaron una disminución en el sacrificio de bovinos debido a los incesantes racionamientos eléctricos.\nAsí lo informó Daniel Ariza, presidente de la Asociación de Ganaderos y Agricultores del municipio Colón (Aganaco), quien aseguró que la medida se tomó en los frigoríficos: Frigorifico Industrial Sur del Lago (Frisulca), Frigorífico Catatutmo (Fricasa) y Frigorífico Industrial Santa Bárbara (Fibasa), para evitar acumulación de reses en canal. “Solo se ejecuta matanza en algunas unidades de producción y los índices descienden con el paso de los días”.\nPrecisó que aun cuando algunos frigoríficos cuentan con plantas eléctricas, estos carecen de combustible, transporte y logística para mantenerlas encendidas todos los días, tal como lo ameritan los cárnicos en cadena de frío. Solo hasta el 8 de abril, Corpoelec diseñó un plan que ha divulgado a través de las redes sociales sin vocería del Gobierno y donde incluye a los 12 circuitos de la subregión en bloques de hasta seis horas diarias de racionamiento.\n“Muchos de los ganaderos dejan de matar animales a raíz de esta situación tan difícil; fue una manera de controlar la pérdida de carne de res, de cerdo y otros animales que acá se producen, manteniéndolos vivos y destinarlos al sacrificio solo cuando Corpoelec asome una posible estabilidad en los circuitos”.\nTambién en las carnicerías optaron por bajar la compra de cárnicos. Los alimentos en estos expendios tienden a perderse con más facilidad, porque esta cadena de comercialización no ha invertido en un plan alternativo para mantener la vida útil de las cavas o cuartos fríos.\nPara el primer apagón, los carniceros se valieron de amigos y productores que resguardaron parte de la carne. Otra la vendieron a bajo costo y una minoría regaló la disponibilidad en las exhibidoras.\nMientras esto ocurre, la carne de res en las expendedoras locales disminuyó y con ello el consumo de proteína animal en los hogares venezolanos. Pero no solo se trata de la carne: la medida incluye los huesos, vísceras o las piezas duras como piel y cuernos del animal.\nLa producción de leche también va en retroceso y descendió desde marzo. Armando Chacín, presidente de la Federación Nacional de Ganaderos de Venezuela (Fedenaga), indicó a\xa0El Pitazo\xa0que durante los primeros cinco días del primer apagón, registrado por un incidente en Guri el 7 de marzo reciente, la pérdida de producción diaria nacional de leche cruda se ubicó en un millón 150.000 litros por día. Sin embargo, luego de tomarse “medidas urgentes”, el gremio logró reducir la cifra.\nUn litro de leche a puerta de planta es cancelado en mil bolívares. Según la federación,\xa0la tercera parte de la producción de leche es destinada a derivados como chicha, choco y demás subproductos que no ameritan pasteurización. Manifestó que quesos y demás preparaciones dependen del lácteo que de igual manera ameritan los neonatos en fase de crecimiento.\nDijo que los 120.000 productores que arriman el lácteo a las distintas manufactureras, procesadoras, pasteurizadoras y queseras debieron paralizar una parte del ordeño en sus unidades de producción.\n“Se originó una situación imprevista. Buena parte del rebaño nacional dejó de ser ordeñado y las vacas fueron destinadas a amamantar becerros.\nNuestra preocupación es que estos animales se van a secar porque les están dando la leche a sus crías y esto amerita que el ordeño se vuelva a realizar hasta un nuevo parto, en un lapso de nueve meses”, estimó Chacín.\nEsta organización resumió las consecuencias de estar tantas horas sin luz en el sector agropecuario de la región en la pérdida de 100.000 litros de leche. Aseguran que el sector ha sido uno de los más golpeados por la falta de electricidad.\nNo se habían recuperado de las consecuencias del primer apagón cuando el segundo interrumpió la cadena de producción. Sin fluido eléctrico es imposible enfriar y mantener en buen estado la leche; las pocas empresas que poseen plantas no operan sin combustible, elemento muy escaso en el municipio Machiques, en donde se dificulta cada día el suministro de gasolina y gasoil. Como consecuencia de esta escasez, también se dificulta el transporte de leche, y por ende la elaboración del queso y su comercialización."
