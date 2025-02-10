import unittest

from cross_db_benchmark.benchmark_tools.dbms.utils import remove_cast_nesting


class UtilsTest(unittest.TestCase):
    def test_remove_cast_nesting(self):
        filter_str = '( uplny_naz=Rozhodnutí vlády o souhlasu s přelety a průjezdy ozbrojených sil členských států Organizace Severoatlantické smlouvy (NATO) a států zúčastněných v programu Partnerství pro mír (PfP) a s přelety ozbrojených sil Bosny a Hercegoviny, Srbska a Černé Hory, Státu Izrael, Jordánského hášimovského království, Egyptské arabské republiky, Království Saúdské Arábie, Státu Kuvajt, Ománu, Spojených arabských emirátů, Bahrajnského království, Syrské arabské republiky, Pákistánské islámské republiky, Alžírské demokratické a lidové republiky, Tuniské republiky, Čadské republiky a Organizace Severoatlantické smlouvy (NATO) přes území České republiky v době od 1. ledna do 31. prosince 2004, na které se vztahuje rozhodovací pravomoc vlády ve smyslu čl. 43 odst. 5 písm. a) ústavního zákona č. 1/1993 Sb., Ústava České republiky, ve znění ústavního zákona č. 300/2000 Sb. AND uplny_naz IS NOT NULL )'
        result = remove_cast_nesting(filter_str)[0]

        self.assertEqual(filter_str, result)


if __name__ == '__main__':
    unittest.main()
