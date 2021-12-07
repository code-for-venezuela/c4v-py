# This is a sample script you can use to test some things by hand, it's just a shortcut and should not be used 
# as an actual testing tool, is just so you can have a list of sample command invocations
# to check for results

echo "-- Testing generic commands >>"
c4v --help          &&

echo "-- Testing listing >>  "  &&
c4v list                        &&
c4v list --urls                 &&
c4v list --limit 7              &&
c4v list --col-len 23           &&
c4v list --count                &&
c4v list --scraped-only true    &&
c4v list --scraped-only false   &&
c4v list --limit 7 --col-len 23 &&
c4v list --col-len 23 --urls    && 
c4v list --limit 7 --count

echo "-- Testing show >> "      &&
c4v show https://primicia.com.ve/nacion/regresan-79-venezolanos-desde-brasil-con-el-plan-vuelta-a-la-patria/             &&
c4v show --no-scrape https://primicia.com.ve/nacion/regresan-79-venezolanos-desde-brasil-con-el-plan-vuelta-a-la-patria/ &&

echo "-- Testing experiments >> "
c4v experiment ls               


echo "-- Testing crawl >> " 
c4v crawl --list                            &&
c4v crawl --limit 10  primicia              &&
c4v crawl --limit 10 --all                  &&
c4v crawl --limit 10 --all-but el_pitazo    &&
c4v crawl --limit 10 --loud primicia  

echo "-- Testing scrape >>"
c4v scrape --limit 10 