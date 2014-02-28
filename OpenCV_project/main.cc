#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define F_BRISK 0
#define F_BRIEF 1
#define F_Freak 2

using namespace cv;
using namespace std;


// Function realizing BRIEF descriptor matching. It takes two input images,
// and four vectors - for outcoming keypoints and descriptors obtained
// from both the images. Keypoints are extracted with FAST algorithm.
void match_BRIEF( cv::Mat& img1, 
                  cv::Mat& img2, 
                  std::vector<cv::KeyPoint>& keypoints1,
                  std::vector<cv::KeyPoint>& keypoints2,
                  cv::Mat& descriptors1, 
                  cv::Mat& descriptors2 
                )
{
   cv::FastFeatureDetector FAST(20);
   cv::BriefDescriptorExtractor BRIEF_extractor(32); // 32 bytes length

   FAST.detect( img1, keypoints1 );
   FAST.detect( img2, keypoints2 );

   BRIEF_extractor.compute( img1, keypoints1, descriptors1);
   BRIEF_extractor.compute( img2, keypoints2, descriptors2);
}

// Function realizing BRISK descriptor matching. It takes two input images,
// and four vectors - for outcoming keypoints and descriptors obtained
// from both the images. Keypoints are extracted with BRISKs
// own feature extranction algorithm.
void match_BRISK( cv::Mat& img1, 
                  cv::Mat& img2, 
                  std::vector<cv::KeyPoint>& keypoints1,
                  std::vector<cv::KeyPoint>& keypoints2,
                  cv::Mat& descriptors1, 
                  cv::Mat& descriptors2 
                )
{
   //set brisk parameters 
   int Threshl=10;
   int Octaves=4; // (pyramid layer) from which the keypoint has been extracted
   float PatternScales=1.0f;
   
   cv::BRISK  BRISKD(Threshl,Octaves,PatternScales);
   BRISKD.create("Feature2D.BRISK");
   
   BRISKD.detect(img1, keypoints1);
   BRISKD.compute(img1, keypoints1, descriptors1);

   BRISKD.detect(img2, keypoints2);
   BRISKD.compute(img2, keypoints2, descriptors2);
}



// Function realizing Freak descriptor matching. It takes two input images,
// and four vectors - for outcoming keypoints and descriptors obtained
// from both the images. Keypoints are extracted with FAST algorithm.
void match_Freak( cv::Mat& img1, 
                  cv::Mat& img2, 
                  std::vector<cv::KeyPoint>& keypoints1,
                  std::vector<cv::KeyPoint>& keypoints2,
                  cv::Mat& descriptors1, 
                  cv::Mat& descriptors2 
                )
{
   cv::FastFeatureDetector FAST(20);
   cv::FREAK Freak_extractor;

   FAST.detect( img1, keypoints1 );
   FAST.detect( img2, keypoints2 );

   Freak_extractor.compute( img1, keypoints1, descriptors1);
   Freak_extractor.compute( img2, keypoints2, descriptors2);
}



void desrciptor_matching( const char * directory, 
                          const char * file1, 
                          const char * file2, 
                          ofstream& report_file, 
                          vector<float>& h,
                          short descriptor_flag )
{
   string result_pic_name;

   // Preparing the resulting pictures paths.
   if( descriptor_flag == F_BRISK )
      result_pic_name = 
         string("datasets/") + 
         string(directory) + 
         string("BRISK_") + 
         string(file1).substr(0,4) + 
         string("_and_") + 
         string(file2).substr(0,4);
   if( descriptor_flag == F_BRIEF )
      result_pic_name = 
         string("datasets/") + 
         string(directory) + 
         string("BRIEF_") + 
         string(file1).substr(0,4) + 
         string("_and_") + 
         string(file2).substr(0,4);
   if( descriptor_flag == F_Freak )
      result_pic_name = 
         string("datasets/") + 
         string(directory) + 
         string("Freak_") + 
         string(file1).substr(0,4) + 
         string("_and_") + 
         string(file2).substr(0,4);
   
   // Preparing input file paths.
   string file_name1 = string("datasets/") + string(directory) + string(file1);
   string file_name2 = string("datasets/") + string(directory) + string(file2);

   // Images have to be loaded in the grayscale.
   cv::Mat GrayA =cv::imread(file_name1.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
   cv::Mat GrayB =cv::imread(file_name2.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

   std::vector<cv::KeyPoint> keypointsA, keypointsB;

   cv::Mat descriptorsA, descriptorsB;

   if ( descriptor_flag == F_BRISK )
      match_BRISK( GrayA, GrayB, keypointsA, keypointsB, descriptorsA, descriptorsB );
   if ( descriptor_flag == F_BRIEF )
      match_BRIEF( GrayA, GrayB, keypointsA, keypointsB, descriptorsA, descriptorsB );
   if ( descriptor_flag == F_Freak )
      match_Freak( GrayA, GrayB, keypointsA, keypointsB, descriptorsA, descriptorsB );

   // Flann matcher is constructing to deal with the Hamming distance as the
   // distance measure. For this purpose it is parametrized with the LshIndexParams
   // function.
   cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20,10,2));

   std::vector<cv::DMatch> matches;
   matcher.match(descriptorsA, descriptorsB, matches);

   int max_distance = 0;
   int min_distance = 1024;

   // Get the minimum and maximum distance in matching task.
   for( int i=0; i < matches.size(); i++)
   {
      if( matches[i].distance > max_distance )
         max_distance = matches[i].distance;

      if( matches[i].distance < min_distance )
         min_distance = matches[i].distance;
   }
  
   // Image with all matches
   cv::Mat all_matches;

   Point2f original;
   Point2f transformed_original;
   Point2f matched;
   double X, Y, Z;
   double d;

   std::vector<cv::DMatch> good_matches;

   // Determine whether the match is correct basing on the information contained in the
   // homography matrix between the images. Some error in the matching is accepted,
   // and the norm of the difference of the transformed original point and the point in the
   // distorted image is acceptable to be not greater than one. In theory, it should be as near
   // to zero as possible.
   for( int i=0; i < matches.size() ; i++)
   {
      original = keypointsA[ matches[i].queryIdx ].pt;
      matched = keypointsB[ matches[i].trainIdx ].pt;

      X = h[0]*original.x + h[1]*original.y + h[2];
      Y = h[3]*original.x + h[4]*original.y + h[5];
      Z = h[6]*original.x + h[7]*original.y + h[8];

      transformed_original.x = X/Z;
      transformed_original.y = Y/Z;

      d = sqrt( (matched.x - transformed_original.x)*(matched.x - transformed_original.x) + 
                (matched.y - transformed_original.y)*(matched.y - transformed_original.y)    );

      if( d < 1 )
         good_matches.push_back(matches[i]);
   }

   // Plot results of selecting good matches based on the homography information.
   cv::drawMatches( GrayA, keypointsA, GrayB, keypointsB,
                        good_matches, all_matches, cv::Scalar(0,255,0), cv::Scalar(0,0,255),
                        vector<char>(),cv::DrawMatchesFlags::DEFAULT );

   IplImage* outrecog = new IplImage(all_matches);
   cvSaveImage( string( result_pic_name + ".ppm" ).c_str(), outrecog );


   std::cout << result_pic_name << "      Done!"<< std::endl; 

   // Write important statistics to the report file.
   report_file << "   Min. distance: "
                << min_distance
                << "\n   Max. distance: "
                << max_distance
                << "\n   Total matches: "
                << matches.size()
                << "\n   Good matches: "
                << good_matches.size()
                << "\n" << endl;
}



int main( int argc, char** argv )
{
   // All directories containing data sets.
   std::vector<const char *> directories;
   directories.push_back("bark/");
   directories.push_back("bikes/");
   directories.push_back("boat/");
   directories.push_back("graf/");
   directories.push_back("leuven/");
   directories.push_back("trees/");
   directories.push_back("ubc/");
   directories.push_back("wall/");
   
   // All files within a data set.
   std::vector<const char *> file_names;
   file_names.push_back("img1.ppm");
   file_names.push_back("img2.ppm");
   file_names.push_back("img3.ppm");
   file_names.push_back("img4.ppm");
   file_names.push_back("img5.ppm");
   file_names.push_back("img6.ppm");
   file_names.push_back("img1.pgm");
   file_names.push_back("img2.pgm");
   file_names.push_back("img3.pgm");
   file_names.push_back("img4.pgm");
   file_names.push_back("img5.pgm");
   file_names.push_back("img6.pgm");

   // All homographies are held in a map structure.
   // The hash is a string build of <directory>+<name> of a certain
   // image file. All homographies describe the transformation of
   // any image file in a certain data set with respect to the img1.*.
   std::map< std::string, std::vector<float> > homographies;

   std::fstream data_file;

   data_file.open("",std::ios_base::in);
   vector<float> vect;
   float a;

   // Loading all homographies matrices and building the structure of the mentioned above map.
   for( int i = 0; i < directories.size(); i++)
   {
      if( string(directories[i]).compare(string("boat/")) == 0 )
      {
         for( int j = 7; j < file_names.size(); j++)
         {
            data_file.open( string( string("datasets/") + directories[i] + string("H1to") + string(file_names[j]).substr(3,1) + string("p") ).c_str(), std::ios_base::in);
            while( data_file >> a )
               vect.push_back(a);
            homographies.insert( std::pair<std::string, std::vector<float> >( string(string(directories[i])+string(file_names[j])) ,vect) );
            vect.clear();
            data_file.close();
         }
      }
      else
      {
         for( int j = 1; j < 6; j++)
         {
            data_file.open( string( string("datasets/") + directories[i] + string("H1to") + string(file_names[j]).substr(3,1) + string("p") ).c_str(), std::ios_base::in);
            while( data_file >> a )
               vect.push_back(a);
            homographies.insert( std::pair<std::string, std::vector<float> >( string(string(directories[i])+string(file_names[j])) ,vect) );
            vect.clear();
            data_file.close();
         }
      }
   }

   ofstream BRISK_report;
   ofstream BRIEF_report;
   ofstream Freak_report;
   BRISK_report.open("BRISK_report.txt");
   BRIEF_report.open("BRIEF_report.txt");
   Freak_report.open("Freak_report.txt");

   // Perform the actual matching between images using each of the three descriptor
   // building algorithms.
   for( int i = 0; i < directories.size(); i++)
   {
      BRISK_report << directories[i] << " data set: ===============" << endl;
      BRIEF_report << directories[i] << " data set: ===============" << endl;
      Freak_report << directories[i] << " data set: ===============" << endl;

      if( string(directories[i]).compare(string("boat/")) == 0 )
      {
         for( int j = 7; j < file_names.size(); j++)
         {
            desrciptor_matching( directories[i], 
                                 file_names[6], 
                                 file_names[j], 
                                 BRISK_report, 
                                 homographies[ string( string(directories[i]) + string(file_names[j]) ) ],
                                 F_BRISK 
            );
            desrciptor_matching( directories[i], 
                                 file_names[6], 
                                 file_names[j], 
                                 BRIEF_report, 
                                 homographies[ string( string(directories[i]) + string(file_names[j]) ) ],
                                 F_BRIEF 
            );
            desrciptor_matching( directories[i], 
                                 file_names[6], 
                                 file_names[j], 
                                 Freak_report, 
                                 homographies[ string( string(directories[i]) + string(file_names[j]) ) ],
                                 F_Freak 
            );
         }
      }
      else
      {
         for( int j = 1; j < 6; j++)
         {
            desrciptor_matching( directories[i], 
                                 file_names[0], 
                                 file_names[j], 
                                 BRISK_report, 
                                 homographies[ string( string(directories[i]) + string(file_names[j]) ) ],
                                 F_BRISK 
            );
            desrciptor_matching( directories[i], 
                                 file_names[0], 
                                 file_names[j], 
                                 BRIEF_report, 
                                 homographies[ string( string(directories[i]) + string(file_names[j]) ) ],
                                 F_BRIEF 
            );
            desrciptor_matching( directories[i], 
                                 file_names[0], 
                                 file_names[j], 
                                 Freak_report, 
                                 homographies[ string( string(directories[i]) + string(file_names[j]) ) ],
                                 F_Freak 
            );
         }
      }

      cout << "\n\n" << directories[i] << "   dataset succesfully computed!" << endl;
   }


   BRISK_report.close();
   BRIEF_report.close();
   Freak_report.close();
   return 0;
}
