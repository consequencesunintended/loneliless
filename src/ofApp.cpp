#include "ofApp.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


//--------------------------------------------------------------
void ofApp::setup() {
	//vagRounded.load( "vag.ttf", 32 );

	m_ball_origin = m_ball_position;
	m_ball_original_direction = m_ball_direction;
	resetLevel();


	Py_SetProgramName( (wchar_t*)"PYTHON" );

	py_test = py::module::import( "py_test" );


}

//--------------------------------------------------------------
void ofApp::update() {
	float dt = (float)ofGetElapsedTimeMillis() / 1000.0f;
	ofResetElapsedTimeCounter();
	dt += 1.0f;

	ofVec2f new_ball_position = m_ball_position + m_ball_direction * dt * 2.0f;

	if ( new_ball_position.y <= 0 )
	{
		new_ball_position.y = 0;
		m_ball_direction.y *= -1;
	}

	else if ( new_ball_position.y >= WIDTH_RES - m_ball_size )
	{
		new_ball_position.y = WIDTH_RES - m_ball_size;
		m_ball_direction.y *= -1;
	}

	else if ( hasCollidedWithPlayer( m_ball_position, new_ball_position ) )
	{
		new_ball_position.x = m_player_position.x;
		m_ball_direction.x *= -1;
		return;
	}

	else if ( new_ball_position.x <= 0 )
	{
		resetLevel();
		new_ball_position = m_ball_position;
	}
	else if ( new_ball_position.x >= HEIGHT_RES - m_ball_size )
	{
		new_ball_position.x = HEIGHT_RES - m_ball_size;
		m_ball_direction.x *= -1;
	}

	m_ball_position = new_ball_position;
	static int times = 1;
}
void scale_by_2( Eigen::Ref<Eigen::VectorXd> v ) {
	v *= 2;
}



//--------------------------------------------------------------
void ofApp::draw() {

	ofBackground( ofColor::black );

	ofSetHexColor( 0xffffff );
	//vagRounded.drawString( std::to_string( m_elapsed_time / 1000 ), 70, 70 );


	ofFill();
	ofDrawRectangle( m_player_position.x, m_player_position.y, m_ball_size, m_player_length );

	ofDrawRectangle( m_ball_position.x, m_ball_position.y, m_ball_size, m_ball_size );

	ofImage screenTemp;
	screenTemp.grabScreen( 0, 0, WIDTH_RES, HEIGHT_RES );

	int m = 2;

	screen[3] = std::move( screen[2] );
	screen[2] = std::move( screen[1] );
	screen[1] = std::move(screen[0]);
	screen[0] = std::move(screenTemp);

	//unsigned char* test = screen[0].getPixels().getData();
	//py::object result = py_test.attr( "get_action" )();


	//int casted_result = result.cast<int>();

	//cout << casted_result << endl;
}

bool ofApp::hasCollidedWithPlayer( const ofVec2f& ball_current_position, const ofVec2f& ball_new_position )
{
	ofVec2f interSection;

	ofVec2f player_position_start = m_player_position;
	ofVec2f player_position_end = player_position_start;

	player_position_end.y += m_player_length;

	return ofLineSegmentIntersection<ofVec2f>( player_position_start, player_position_end, ball_current_position, ball_new_position, interSection );
}

//--------------------------------------------------------------
void ofApp::keyPressed( int key ) {

	float dt = (float)ofGetElapsedTimeMillis() / 1000.0f;
	ofResetElapsedTimeCounter();

	if ( !m_key_pressed )
	{
		dt = 0;
		m_key_pressed = true;

	}
	dt += 1.0f;

	float dx = 5.0f * dt;

	if ( key == OF_KEY_UP )
	{
		if ( m_player_position.y - dx >= 0 )
		{
			m_player_position.y -= dx;
		}
		else
		{
			m_player_position.y = 0;
		}
	}
	else if ( key == OF_KEY_DOWN )
	{
		int height = ofGetWindowHeight();

		if ( m_player_position.y + m_player_length + dx <= height )
		{
			m_player_position.y += dx;
		}
		else
		{
			m_player_position.y = height - m_player_length;
		}
	}
	else if ( key == OF_KEY_F10 )
	{
		resetLevel();
	}
}

void ofApp::resetLevel()
{
	m_ball_position = m_ball_origin;
	m_ball_direction = m_ball_original_direction;
	m_ball_direction.rotate( ofRandom( 90 ) + 45 );
}

//--------------------------------------------------------------
void ofApp::keyReleased( int key ) {

	m_key_pressed = false;
}

//--------------------------------------------------------------
void ofApp::mouseMoved( int x, int y ) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged( int x, int y, int button ) {

}

//--------------------------------------------------------------
void ofApp::mousePressed( int x, int y, int button ) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased( int x, int y, int button ) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered( int x, int y ) {

}

//--------------------------------------------------------------
void ofApp::mouseExited( int x, int y ) {

}

//--------------------------------------------------------------
void ofApp::windowResized( int w, int h ) {

}

//--------------------------------------------------------------
void ofApp::gotMessage( ofMessage msg ) {

}

//--------------------------------------------------------------
void ofApp::dragEvent( ofDragInfo dragInfo ) {

}
