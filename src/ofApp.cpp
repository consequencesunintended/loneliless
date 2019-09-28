#include "ofApp.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

//--------------------------------------------------------------
void ofApp::setup() {

	m_ball_origin = m_ball_position;
	m_ball_original_direction = m_ball_direction;
	resetLevel();


	Py_SetProgramName( (wchar_t*)"PYTHON" );

	// make sure the data directory is been added to python path, so 
	// .py files can be loaded from the default data folder
	const std::string&	data_directory = ofToDataPath( "", true );	
	py::module			sys = py::module::import( "sys" );

	sys.attr( "path" ).attr( "insert" )(1, data_directory);

	// import dqn.py 
	m_dqn_module = py::module::import( "dqn" );

	// set the default location for saving and restoring the model variables
	const std::string&	model_directory_string = data_directory + "/model";
	const std::string&	model_variables_path = model_directory_string + "/loneliless.ckpt";

	m_dqn_module.attr( "setSavedModelPath" )(model_variables_path);

	py::object			m_num_of_frames_to_buffer_value = m_dqn_module.attr( "getNumFramesToStore" )();

	m_num_of_frames_to_buffer = m_num_of_frames_to_buffer_value.cast<int>();

	ofDirectory			model_directory( model_directory_string );

	if ( model_directory.exists() && model_directory.getFiles().size() )
	{
		if ( m_game_mode == GAMEMODE::AI_RESTORE_MODE )
		{
			m_dqn_module.attr( "restoreMode" )();
		}
	}
	else
	{
		model_directory.create();
		m_game_mode = GAMEMODE::AI_TRAIN_MODE;
	}
}

//--------------------------------------------------------------
void ofApp::update() {
	float dt = (float)ofGetElapsedTimeMillis() / 1000.0f;
	ofResetElapsedTimeCounter();
	dt += 1.0f;

	if ( ( m_game_mode == GAMEMODE::AI_TRAIN_MODE ) || ( m_game_mode == GAMEMODE::AI_RESTORE_MODE ) )
	{
		if ( m_current_frame == 0 )
		{
			py::object action_pyvalue;

			if ( m_game_mode == GAMEMODE::AI_TRAIN_MODE )
			{
				action_pyvalue = m_dqn_module.attr( "get_action" )();
			}
			else
			{
				action_pyvalue = m_dqn_module.attr( "get_trained_action" )();
			}
			m_action = action_pyvalue.cast<int>();
		}

		if ( !m_done )
		{
			if ( m_action == 0 )
			{
				moveUp( dt );
			}
			else if ( m_action == 2 )
			{
				moveDown( dt );
			}
			float temp_reward;

			updateBallPosition( dt, m_retflag, m_done, temp_reward );

			m_reward += temp_reward;
		}

		if ( m_current_frame == 0 )
		{
			ofImage screenTemp;

			screenTemp.grabScreen( 0, 0, WIDTH_RES, HEIGHT_RES );
			screenTemp.setImageType( OF_IMAGE_GRAYSCALE );

			auto frame = Eigen::Map<Eigen::Matrix<unsigned char, WIDTH_RES, HEIGHT_RES > >( screenTemp.getPixels().getData() );

			if ( !m_initial_frames_set )
			{
				for ( int i = 0; i < m_num_of_frames_to_buffer - 1; i++ )
				{
					m_dqn_module.attr( "buffer_frame" )(frame);
				}
				m_initial_frames_set = true;
			}
			m_dqn_module.attr( "buffer_frame" )(frame);

			if ( m_game_mode == GAMEMODE::AI_TRAIN_MODE )
			{
				m_dqn_module.attr( "add_replay_memory" )(m_action, m_reward, m_done);
			}

			m_reward = 0;

			if ( m_done )
			{
				resetLevel();
			}
		}
		m_current_frame++;

		if ( m_current_frame == m_frames_to_skip )
		{
			m_current_frame = 0;
		}
	}
	else if ( m_game_mode == GAMEMODE::PLAYER_MODE )
	{
		float temp_reward;
		updateBallPosition( dt, m_retflag, m_done, temp_reward );

		if ( m_done )
		{
			resetLevel();
		}
	}
}

void ofApp::updateBallPosition( float dt, bool& retflag, bool& done, float& reward )
{
	retflag = true;
	done = false;
	reward = 0.0f;
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
		reward = 1.0f;
		new_ball_position.x = m_player_position.x;
		m_ball_direction.x *= -1;
		return;
	}

	else if ( new_ball_position.x <= 0 )
	{
		reward = -1.0f;
		done = true;
		new_ball_position = m_ball_position;
	}
	else if ( new_ball_position.x >= HEIGHT_RES - m_ball_size )
	{
		new_ball_position.x = HEIGHT_RES - m_ball_size;
		m_ball_direction.x *= -1;
	}

	m_ball_position = new_ball_position;
	retflag = false;
}


//--------------------------------------------------------------
void ofApp::draw() {

	ofBackground( ofColor::black );

	ofSetHexColor( 0xffffff );


	ofFill();
	ofDrawRectangle( m_player_position.x, m_player_position.y, m_ball_size, m_player_length );

	ofDrawRectangle( m_ball_position.x, m_ball_position.y, m_ball_size, m_ball_size );

}

bool ofApp::hasCollidedWithPlayer( const ofVec2f& ball_current_position, const ofVec2f& ball_new_position )
{
	ofVec2f interSection;

	ofVec2f player_position_start = m_player_position;
	ofVec2f player_position_end = player_position_start;

	player_position_end.y += m_player_length + m_ball_size;
	player_position_start.y -= m_ball_size;
	return ofLineSegmentIntersection<ofVec2f>( player_position_start, player_position_end, ball_current_position, ball_new_position, interSection );
}

void ofApp::moveUp( float dx )
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

//--------------------------------------------------------------
void ofApp::keyPressed( int key ) {\

	float dt = (float)ofGetElapsedTimeMillis() / 1000.0f;
	ofResetElapsedTimeCounter();

	if ( !m_key_pressed )
	{
		dt = 0;
		m_key_pressed = true;

	}
	dt += 1.0f;

	float dx = 5.0f * dt;

	if ( m_game_mode == GAMEMODE::PLAYER_MODE )
	{
		if ( key == OF_KEY_UP )
		{
			moveUp( dx );
		}
		else if ( key == OF_KEY_DOWN )
		{
			moveDown( dx );
		}
	}

	if ( key == OF_KEY_F10 )
	{
		resetLevel();
	}
}

void ofApp::moveDown( float dx )
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

void ofApp::resetLevel()
{
	m_player_position = ofVec2f( 10.0f, ofRandom( 10, HEIGHT_RES - 10 ) );
	m_ball_position = m_ball_origin;
	m_ball_direction = m_ball_original_direction;
	m_ball_direction.rotate( ofRandom( 40 ) + 32 );
	m_done = false;
	m_current_frame = 0;
	m_initial_frames_set = false;
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
