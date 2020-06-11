#include "ofMain.h"
#include "ofApp.h"


//========================================================================
int main() {

#if	DRAW_DEBUG_IMAGES
	ofSetupOpenGL( WIDTH_RES + 320, HEIGHT_RES, OF_WINDOW );			// <-------- setup the GL context
#else
	ofSetupOpenGL( WIDTH_RES, HEIGHT_RES, OF_WINDOW );
#endif // DRAW_DEBUG_IMAGES

	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN
	// pass in width and height too:
	ofRunApp( new ofApp() );

}
