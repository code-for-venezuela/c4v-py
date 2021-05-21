from c4v.scraper.crawler.crawlers.el_pitazo_crawler import ElPitazoCrawler
from c4v.scraper.crawler.crawlers.primicia_crawler import PrimiciaCrawler


def test_crawler_parse_el_pitazo_index():
    elp = ElPitazoCrawler()

    out = elp.parse_sitemaps_urls_from_index(el_pitazo_sitemap_index())
    ans = ["https://elpitazo.net/sitemap-pt-post-2021-05.xml", "https://elpitazo.net/sitemap-pt-post-2021-04.xml", "https://elpitazo.net/sitemap-pt-post-2021-03.xml"]

    assert set(out) == set(ans)

def test_crawler_parse_el_pitazo_sitemap():
    elp = ElPitazoCrawler()

    out = elp.parse_urls_from_sitemap(el_pitazo_sitemap_body())
    ans = ["https://elpitazo.net/gran-caracas/el-pitazo-sigue-en-pie-para-vencer-la-censura-y-el-bloqueo/", "https://elpitazo.net/cultura/ozuna-anuncia-concierto-virtual-gratuito-en-youtube/", "https://elpitazo.net/politica/almagro-insiste-en-seguir-denunciando-crimenes-de-lesa-humanidad/"]

    assert set(out) == set(ans)

def test_crawler_parse_primicia_index():
    prm = PrimiciaCrawler()

    out = prm.parse_sitemaps_urls_from_index(primicia_sitemap_index())
    ans = ['https://primicia.com.ve/post-sitemap1.xml', 'https://primicia.com.ve/post-sitemap2.xml', 'https://primicia.com.ve/post-sitemap3.xml']

    assert set(out) == set(ans)

def test_crawler_parse_primicia_sitemap():
    prm = PrimiciaCrawler()

    out = prm.parse_urls_from_sitemap(primicia_sitemap_body())
    ans = ["https://primicia.com.ve/sucesos/un-muerto-y-cuatro-heridos-durante-protestas/", "https://primicia.com.ve/mas/salud/the-lancet-global-health-aumentan-muertes-infantiles-en-venezuela/"]
    
    assert set(out) == set(ans)

def el_pitazo_sitemap_index():
    # TODO move to resource folder
    return '<?xml version="1.0" encoding="UTF-8"?><?xml-stylesheet type="text/xsl" href="/wp-content/plugins/google-sitemap-generator/sitemap.xsl"?><!-- sitemap-generator-url="http://www.arnebrachhold.de" sitemap-generator-version="4.1.1" -->\
<!-- generated-on="21 May, 2021 7:53 pm 7:53 pm" -->\
<sitemapindex xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/siteindex.xsd" xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">     <sitemap>\
                <loc>https://elpitazo.net/sitemap-misc.xml</loc>\
                <lastmod>2021-05-21T19:51:20+00:00</lastmod>\
        </sitemap>\
        <sitemap>\
                <loc>https://elpitazo.net/sitemap-pt-post-2021-05.xml</loc>\
                <lastmod>2021-05-21T19:51:20+00:00</lastmod>\
        </sitemap>\
        <sitemap>\
                <loc>https://elpitazo.net/sitemap-pt-post-2021-04.xml</loc>\
                <lastmod>2021-05-03T19:44:49+00:00</lastmod>\
        </sitemap>\
        <sitemap>\
                <loc>https://elpitazo.net/sitemap-pt-post-2021-03.xml</loc>\
                <lastmod>2021-04-23T18:52:36+00:00</lastmod>\
        </sitemap>\
</sitemapindex><!-- Request ID: 1035a4e5fdc69346cc9345dfcde625e9; Queries for sitemap: 5; Total queries: 67; Seconds: 1.14; Memory for sitemap: 0MB; Total memory: 10.00390625MB -->'

def el_pitazo_sitemap_body():
    return '<?xml version="1.0" encoding="UTF-8"?><?xml-stylesheet type="text/xsl" href="/wp-content/plugins/google-sitemap-generator/sitemap.xsl"?><!-- sitemap-generator-url="http://www.arnebrachhold.de" sitemap-generator-version="4.1.1" -->\
<!-- generated-on="21 May, 2021 9:38 pm 9:38 pm" -->\
<urlset xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd" xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">	<url>\
		<loc>https://elpitazo.net/gran-caracas/el-pitazo-sigue-en-pie-para-vencer-la-censura-y-el-bloqueo/</loc>\
		<lastmod>2021-05-01T01:47:03+00:00</lastmod>\
		<changefreq>monthly</changefreq>\
		<priority>0.2</priority>\
	</url>\
	<url>\
		<loc>https://elpitazo.net/cultura/ozuna-anuncia-concierto-virtual-gratuito-en-youtube/</loc>\
		<lastmod>2021-04-30T23:48:58+00:00</lastmod>\
		<changefreq>monthly</changefreq>\
		<priority>0.2</priority>\
	</url>\
	<url>\
		<loc>https://elpitazo.net/politica/almagro-insiste-en-seguir-denunciando-crimenes-de-lesa-humanidad/</loc>\
		<lastmod>2021-04-30T23:27:55+00:00</lastmod>\
		<changefreq>monthly</changefreq>\
		<priority>0.2</priority>\
	</url>\
</urlset><!-- Request ID: 3808009fddbeb11af0169f7586476ae0; Queries for sitemap: 6602; Total queries: 6661; Seconds: 5.75; Memory for sitemap: 136.6640625MB; Total memory: 146.66796875MB -->'

def primicia_sitemap_index():
    return '<?xml version="1.0" encoding="UTF-8"?><?xml-stylesheet type="text/xsl" href="//primicia.com.ve/wp-content/plugins/wordpress-seo/css/main-sitemap.xsl"?>\
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\
        <sitemap>\
                <loc>https://primicia.com.ve/post-sitemap1.xml</loc>\
                <lastmod>2019-02-15T01:27:26+00:00</lastmod>\
        </sitemap>\
        <sitemap>\
                <loc>https://primicia.com.ve/post-sitemap2.xml</loc>\
                <lastmod>2019-03-07T19:03:35+00:00</lastmod>\
        </sitemap>\
        <sitemap>\
                <loc>https://primicia.com.ve/post-sitemap3.xml</loc>\
                <lastmod>2019-03-26T04:33:30+00:00</lastmod>\
        </sitemap>\
        <sitemap>\
                <loc>https://primicia.com.ve/author-sitemap.xml</loc>\
                <lastmod>2020-04-28T17:47:34+00:00</lastmod>\
        </sitemap>\
        </sitemapindex>'

def primicia_sitemap_index():
    return '<?xml version="1.0" encoding="UTF-8"?><?xml-stylesheet type="text/xsl" href="//primicia.com.ve/wp-content/plugins/wordpress-seo/css/main-sitemap.xsl"?>\
<urlset xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1" xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd http://www.google.com/schemas/sitemap-image/1.1 http://www.google.com/schemas/sitemap-image/1.1/sitemap-image.xsd" xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\
        <url>\
                <loc>https://primicia.com.ve/</loc>\
                <lastmod>2021-05-21T22:04:03+00:00</lastmod>\
        </url>\
        <url>\
                <loc>https://primicia.com.ve/sucesos/un-muerto-y-cuatro-heridos-durante-protestas/</loc>\
                <lastmod>2019-01-25T15:30:32+00:00</lastmod>\
                <image:image>\
                        <image:loc>https://primicia.com.ve/wp-content/uploads/2019/01/protestaas.jpg</image:loc>\
                        <image:title><![CDATA[protestaas]]></image:title>\
                </image:image>\
        </url>\
        <url>\
                <loc>https://primicia.com.ve/mas/salud/the-lancet-global-health-aumentan-muertes-infantiles-en-venezuela/</loc>\
                <lastmod>2019-01-25T15:35:34+00:00</lastmod>\
                <image:image>\
                        <image:loc>https://primicia.com.ve/wp-content/uploads/2019/01/hospital.jpg</image:loc>\
                        <image:title><![CDATA[hospital]]></image:title>\
                </image:image>\
        </url>'

def primicia_sitemap_body():
    return '<?xml version="1.0" encoding="UTF-8"?><?xml-stylesheet type="text/xsl" href="//primicia.com.ve/wp-content/plugins/wordpress-seo/css/main-sitemap.xsl"?>\
<urlset xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1" xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd http://www.google.com/schemas/sitemap-image/1.1 http://www.google.com/schemas/sitemap-image/1.1/sitemap-image.xsd" xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\
	<url>\
		<loc>https://primicia.com.ve/</loc>\
		<lastmod>2021-05-21T22:15:27+00:00</lastmod>\
	</url>\
	<url>\
		<loc>https://primicia.com.ve/sucesos/un-muerto-y-cuatro-heridos-durante-protestas/</loc>\
		<lastmod>2019-01-25T15:30:32+00:00</lastmod>\
		<image:image>\
			<image:loc>https://primicia.com.ve/wp-content/uploads/2019/01/protestaas.jpg</image:loc>\
			<image:title><![CDATA[protestaas]]></image:title>\
		</image:image>\
	</url>\
	<url>\
		<loc>https://primicia.com.ve/mas/salud/the-lancet-global-health-aumentan-muertes-infantiles-en-venezuela/</loc>\
		<lastmod>2019-01-25T15:35:34+00:00</lastmod>\
		<image:image>\
			<image:loc>https://primicia.com.ve/wp-content/uploads/2019/01/hospital.jpg</image:loc>\
			<image:title><![CDATA[hospital]]></image:title>\
		</image:image>\
	</url>\
</urlset>\
<!-- XML Sitemap generated by Yoast SEO -->'