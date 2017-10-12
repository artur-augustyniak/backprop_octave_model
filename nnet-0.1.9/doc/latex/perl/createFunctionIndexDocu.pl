#!/usr/bin/perl -w

use strict;
use diagnostics;# Force verbose warning diagnostics.
use warnings;
use English;

#
# Modules from the Standard Perl Library.


#
# Own modules
use analyzeOctaveSource;


#--- DEFINE VARIABLES -------------------------------
my $Dir = "D:/daten/octave/neuroPackage/0.1.9/nnet/inst";
my $fileExt = "m";
my $funcIndDir = "D:\\daten\\octave\\neuroPackage\\documentation\\latex\\developers\\funcindex";
my $relFuncDir = "funcindex/";
my $funcFileExt = "tex";
my $mainLatexFuncIndexFile = "funcindex.tex";
my $chapter = "Function Index";
#--- END DEFINE VARIABLES ---------------------------


my @filesArray = ();
my @filesName = ();
my $fileName = "";
my $savePath = "";
my $nTestLines = 0;
my $m = 0;

# analyze directory structure
my $obj = analyzeOctaveSource->new();
my @DirArray = $obj->readDirTree($Dir,$fileExt);

# in @DirArray should be only names, ending with *.m
# but if there is a directory, we will remove it at
# the next line of code
my @FuncReferences = grep /.+\.$fileExt/ , @DirArray; #/
my @FuncNames = @FuncReferences;
# now I have to remove the path and file extension
foreach (@FuncNames)
{
 	s/\.m//g;
 	s/$Dir\///g;
	print "$_\n";
#	sleep(1);
}

# now analyze functions to see which other functions are called
foreach my $FuncRef (@FuncReferences)
{
   foreach my $FuncName (@FuncNames)
   {
     if ($FuncRef !~/$FuncName/)   # returns true if pattern is not found
     {
       # ignore all lines starting with # oder %

     }
   }

}
#
# # analyze file structure
# my $nFiles = @DirArray;
# if ($nFiles>=1){ # if $nFiles==0 readDirTree will die
# 
#   # if we are in this brand we have to parse all files
#   # ending with *.m for function reference! As reference
#   # functions, we will take all the functions inside of
#   # nnet/inst
#   
#   #\todo, an dieser stelle muss weiter gearbeitet werden !!!
#   foreach (@DirArray){
# 	  # open file and search for lines beginning with
# 	  # !%
# 	  if (-d $_){
#         # if directory, do nothing
#       }else{
#       	print "$_\n";
#       	sleep(0.1);
#       	
# 	    open(FILE,$_) or die "File $_ not found!\n";
# 	    my @fileContent = <FILE>;
# 	    chomp(@fileContent);
# 	    $nTestLines = @fileContent;
#         my @temp = grep /^%!/, @fileContent;
#         my $nTemp = @temp;
#         if ($nTemp>0){ # this means, 
#                        # test lines are available
#           # now create latex files without header
#           # take the same name like the *.m file
#           # and save in specified directory $testDir
#           # with file extens in $testFileExt
#           # use verbatim environment
#           @filesName = split(/\//, $_);
#           $fileName = $filesName[$#filesName];
#           # now remove file extension .m
#           @filesName = split(/\./,$fileName);
#           $savePath = ("$testDir" . "\\\\" . "$filesName[0]." . "$testFileExt");
#           open(OUTFILE, ">$savePath");
#           my $i = 0;
#           print OUTFILE "\\begin{verbatim}\n";
#           while ($i < $nTestLines){
#             if ($fileContent[$i]=~/^%!/){
#               print OUTFILE "$fileContent[$i]\n";            
#             }
#             $i++;
# 		  } # END while ($i <= $#fileContent)
# 		  print OUTFILE "\\end{verbatim}\n";
# 		  close OUTFILE;
# 		  
# 		  ## now set entries in the main test latex file ..
#           my $mainTestFile = ("$testDir" . "\\\\" . "$mainLatexTestFile");
#           if ($m==0){
#             open(TESTFILE,">$mainTestFile") or die "Could not found file $mainTestFile!\n";
#             print TESTFILE "\\chapter{$chapter}\n\n";
#             
# 			# test if back slash needed
#             # a back slash is needed if the sign "underscore"
#             # is used in the file name. This happens at each
#             # "sub function". There are two underscores!
#             my $tempString = "";
#             my $oldString = $filesName[0];
# 			$_ = $filesName[0];
# 			s/_/\\_/g; # s/ : search & replace pattern (everything between / /)
# 					   # here: search underscore
# 					   # if found, replace with \_ (two back slashes are needed
# 					   # to get \_ as sign)
# 					   # /g means: each occurence of pattern, otherwise, only one _
# 					   # will be replaced
# 
# 
#            #  if ($filesName[0]=~/_/){
# #             	my $pos = 0; #temp Position
# #             	my $posBefore = 0;
# #             	while ($pos < (length $filesName[0]) and $pos != -1){
# #             		$pos = index($filesName[0], "_", $pos);
# #             		if ($pos != -1 ){
# #             			$tempString = substr($filesName[0],$posBefore,$pos) . "\\" . substr($filesName[0],$pos,length $filesName[0]);
# #             			$posBefore = $pos;
# #             			$pos++;
# #             	    }
# #             	}# END while ($pos < ...)
# #             	$filesName[0] = $tempString;
# #             }
# 			print "test file name: $_\n";
#             print TESTFILE "\\section{$_}\n";
#             $tempString = $relTestDir . $oldString;
#             print TESTFILE "\\input{$tempString}\n";
#           }else{
#             open(TESTFILE,">>$mainTestFile") or die "Could not found file $mainTestFile!\n";
#             # test if back slash needed
#             my $tempString = "";
#             my $oldString = $filesName[0];
#             # test if back slash needed
#             # a back slash is needed if the sign "underscore"
#             # is used in the file name. This happens at each
#             # "sub function". There are two underscores!
#             my $tempString = "";
#             my $oldString = $filesName[0];
# 			$_ = $filesName[0];
# 			s/_/\\_/g; # s/ : search & replace pattern (everything between / /)
# 					   # here: search underscore
# 					   # if found, replace with \_ (two back slashes are needed
# 					   # to get \_ as sign)
# 					   # /g means: each occurence of pattern, otherwise, only one _
# 					   # will be replaced
#             # if ($filesName[0]=~/_/){
# #             	my $pos = 0; #temp Position
# #             	my $posBefore = 0;
# #             	while ($pos < (length $filesName[0]) and $pos != -1){
# #             		$pos = index($filesName[0], "_", $pos);
# #             		if ($pos != -1 ){
# #             			$tempString = substr($filesName[0],$posBefore,$pos) . "\\" . substr($filesName[0],$pos,length $filesName[0]);
# #             			$pos++;
# #             	    }
# #             	}# END while ($pos < ...)
# #             	$filesName[0] = $tempString;
# #             }
#             
#             print TESTFILE "\\section{$_}\n";
#             $tempString = $relTestDir . $oldString;
#             print TESTFILE "\\input{$tempString}\n";
#           }
#           $m++;
#           close TESTFILE;
#   
# 		  
#         }# END if($nTemp>0)
#         close FILE;
#                
# 	  }# END if(-d $_)
# 	  
# 	}# END foreach (@DirArray)
# 
# }else{ # if $nFiles==0
#   print "No file found with valid file extension: .$fileExt.\n";
#   die;
# }